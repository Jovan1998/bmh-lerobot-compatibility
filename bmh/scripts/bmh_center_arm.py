#!/usr/bin/env python
"""
Center Arm — moves every motor of one SO-101 arm to its calibrated center.

Post-calibration sanity check for the BMH-101: with a correct calibration,
"center" puts every body/head joint at the physical middle of its recorded
range of motion (normalized 0) and the gripper at 50 %. A motor that does not
move, or ends far from its target, points at a wiring, ID, or calibration
problem for that joint.

The motors are ramped from their current position to center with a smoothstep
interpolation (no single full-speed goal jump), then hold the pose until ENTER
is received on stdin (or stdin closes) — only then is torque released. The
controller-app drives the release via its "Release Motors" button.

Output is line-oriented and stable so the controller-app can parse it:

    RESULT name=shoulder_pan start=-12.4 target=0.0 end=0.3 ok=true
    ...
    CENTERING_DONE ok=true
    HOLDING press ENTER to release the motors

With `--torque_limit` (0..1000, the servos' `Torque_Limit` RAM register) the arm is
driven weakly: the limit is written to every motor before any torque comes on, the motors
are then enabled one at a time, 50 ms apart, and the previous limits are written back
when the motors are released. This is the bench test for the DAgger leader assist: the
leader's supply is small, so it must be shown that the leader Pi and its USB bus stay up
while the leader arm pulls lightly and can still be overpowered by hand.

Exit codes: 0 = all joints reached center, 2 = some joint missed its target,
1 = error (no calibration, port busy, ...).

Usage:
    bmh-center-arm --type follower --port /dev/ttyACM0 --id so_follower_left --with_head
    bmh-center-arm --type leader --port /dev/ttyACM1 --id so_leader_right
    bmh-center-arm --type leader --port /dev/ttyACM1 --id so_leader_left --with_head --torque_limit 100
"""

import argparse
import contextlib
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Normalized target per norm mode: body/head joints read 0 at the calibrated
# mid-range (DEGREES and RANGE_M100_100 alike), the gripper's RANGE_0_100 reads 50.
CENTER_TOLERANCE = 8.0

# Range of the servos' `Torque_Limit` register (1000 = full torque).
TORQUE_LIMIT_MAX = 1000
# Pause between two motors' torque coming on with `--torque_limit`, so their inrush
# currents do not add up on a weak supply.
TORQUE_STAGGER_S = 0.05


def create_arm(arm_type: str, port: str, arm_id: str, with_head: bool):
    """Instantiate (without connecting) the follower robot or leader teleoperator."""
    if arm_type == "follower":
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        return SOFollower(SOFollowerRobotConfig(port=port, id=arm_id, with_head=with_head))
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    return SOLeader(SOLeaderTeleopConfig(port=port, id=arm_id, with_head=with_head))


def center_targets(bus) -> dict[str, float]:
    """Normalized center position per motor (0 for body/head joints, 50 for the gripper)."""
    from lerobot.motors import MotorNormMode

    return {
        motor: 50.0 if bus.motors[motor].norm_mode is MotorNormMode.RANGE_0_100 else 0.0
        for motor in bus.motors
    }


def smoothstep(t: float) -> float:
    """Ease-in/ease-out interpolation factor for t in [0, 1]."""
    return t * t * (3.0 - 2.0 * t)


def ramp_to(bus, start: dict[str, float], target: dict[str, float], duration: float, fps: int) -> None:
    """Ramps all motors from `start` to `target` over `duration` seconds at `fps` goal writes/s."""
    steps = max(1, int(duration * fps))
    for i in range(1, steps + 1):
        alpha = smoothstep(i / steps)
        goal = {m: start[m] + (target[m] - start[m]) * alpha for m in target}
        bus.sync_write("Goal_Position", goal)
        time.sleep(1.0 / fps)


def limit_torque(bus, torque_limit: int) -> dict[str, int]:
    """Write `torque_limit` to every motor's `Torque_Limit`; return the values it replaces."""
    previous = {motor: int(bus.read("Torque_Limit", motor, normalize=False)) for motor in bus.motors}
    for motor in bus.motors:
        bus.write("Torque_Limit", motor, torque_limit, normalize=False)
    return previous


def enable_torque_staggered(bus, pause_s: float = TORQUE_STAGGER_S) -> None:
    """Turn the torque on one motor at a time, `pause_s` apart."""
    for motor in bus.motors:
        bus.enable_torque(motor)
        time.sleep(pause_s)


def restore_torque_limits(bus, previous: dict[str, int]) -> None:
    """Write the limits `limit_torque` replaced back; a motor that fails is logged, the rest still run."""
    for motor, value in previous.items():
        try:
            bus.write("Torque_Limit", motor, value, normalize=False)
        except Exception:
            logger.warning("Failed to restore Torque_Limit=%d on %s", value, motor, exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move every motor of one SO-101 arm to its calibrated center position"
    )
    parser.add_argument("--type", required=True, choices=["follower", "leader"], help="Arm type")
    parser.add_argument("--port", required=True, help="USB serial port of the arm")
    parser.add_argument("--id", required=True, help="Calibration ID of the arm (e.g. so_follower_left)")
    parser.add_argument(
        "--with_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether this arm's bus carries the head servos (head_pan/head_tilt, IDs 8/9)",
    )
    parser.add_argument("--duration", type=float, default=2.5, help="Ramp duration in seconds")
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Seconds to let the motors settle after the ramp before measuring the end positions",
    )
    parser.add_argument("--fps", type=int, default=30, help="Goal-position write frequency during the ramp")
    parser.add_argument(
        "--torque_limit",
        type=int,
        default=None,
        help=f"Drive the motors with this Torque_Limit (0..{TORQUE_LIMIT_MAX}) instead of full torque, "
        "enabling them one at a time; the previous limits are restored on release",
    )
    args = parser.parse_args()
    if args.torque_limit is not None and not 0 <= args.torque_limit <= TORQUE_LIMIT_MAX:
        parser.error(f"--torque_limit must be in [0, {TORQUE_LIMIT_MAX}]")

    arm = create_arm(args.type, args.port, args.id, args.with_head)

    if not arm.calibration:
        print(
            f"ERROR no calibration file found for id '{args.id}' "
            f"(expected at {arm.calibration_fpath}). Calibrate this arm first.",
            flush=True,
        )
        sys.exit(1)

    # connect(calibrate=False) never triggers the interactive calibration flow.
    # The follower's configure() re-enables torque afterwards; the leader's leaves it off.
    arm.connect(calibrate=False)
    bus = arm.bus
    previous_limits: dict[str, int] = {}
    try:
        if not bus.is_calibrated:
            # Motor registers drifted from the calibration file (e.g. replaced servo):
            # rewrite the file values to the motors, exactly like the calibrate flow does.
            logger.info("Motor calibration registers differ from the file — rewriting them")
            bus.write_calibration(arm.calibration)

        start = bus.sync_read("Present_Position")
        targets = center_targets(bus)

        print(
            f"CENTERING arm={args.id} type={args.type} port={args.port} joints={len(targets)}",
            flush=True,
        )

        # Pin the goal to the present position before enabling torque so no motor jumps.
        bus.sync_write("Goal_Position", start)
        if args.torque_limit is None:
            bus.enable_torque()
        else:
            # The limit first: no motor may ever see full torque, not even for one packet.
            previous_limits = limit_torque(bus, args.torque_limit)
            print(f"TORQUE_LIMIT value={args.torque_limit} motors={len(previous_limits)}", flush=True)
            enable_torque_staggered(bus)
        time.sleep(0.2)

        ramp_to(bus, start, targets, args.duration, args.fps)
        time.sleep(args.settle)

        end = bus.sync_read("Present_Position")
        all_ok = True
        for motor in targets:
            ok = abs(end[motor] - targets[motor]) <= CENTER_TOLERANCE
            all_ok = all_ok and ok
            print(
                f"RESULT name={motor} start={start[motor]:.1f} target={targets[motor]:.1f} "
                f"end={end[motor]:.1f} ok={'true' if ok else 'false'}",
                flush=True,
            )
        print(f"CENTERING_DONE ok={'true' if all_ok else 'false'}", flush=True)

        # Hold the pose under torque until the operator releases it (ENTER over
        # stdin, stdin EOF, or Ctrl-C) — check the arm visually while it holds.
        print("HOLDING press ENTER to release the motors", flush=True)
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            input()
        sys.exit(0 if all_ok else 2)
    finally:
        # Release torque so the arm is free again (leader must never stay stiff);
        # disconnect() also drops torque for the follower via disable_torque_on_disconnect.
        try:
            bus.disable_torque()
        except Exception:
            logger.warning("Failed to disable torque during cleanup", exc_info=True)
        # After the torque is off, so no motor is at full torque while it is still on.
        restore_torque_limits(bus, previous_limits)
        try:
            arm.disconnect()
        except Exception:
            logger.warning("Failed to disconnect cleanly", exc_info=True)


if __name__ == "__main__":
    main()
