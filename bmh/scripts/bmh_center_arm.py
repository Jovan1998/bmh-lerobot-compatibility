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

Exit codes: 0 = all joints reached center, 2 = some joint missed its target,
1 = error (no calibration, port busy, ...).

Usage:
    bmh-center-arm --type follower --port /dev/ttyACM0 --id so_follower_left --with_head
    bmh-center-arm --type leader --port /dev/ttyACM1 --id so_leader_right
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
    args = parser.parse_args()

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
        bus.enable_torque()
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
        try:
            arm.disconnect()
        except Exception:
            logger.warning("Failed to disconnect cleanly", exc_info=True)


if __name__ == "__main__":
    main()
