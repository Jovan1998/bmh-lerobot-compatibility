#!/usr/bin/env python
"""
Read Pose — prints the present pose of both BMH-101 follower arms (and the head).

The controller-app runs this once when the operator saves the robot's Home position:
the operator poses the robot by hand, and this script reads every joint without
moving anything. Only the serial bus is opened (no camera, no `configure()`), torque
is never changed, and the port is closed without touching torque again.

Output is a single line the controller-app parses:

    POSE {"left_shoulder_pan.pos": 1.2, ..., "left_head_tilt.pos": -3.4, "right_gripper.pos": 12.0}

Keys are the bimanual action keys (`left_*` / `right_*`, head on the left bus); values
use the same normalization every lerobot process in the app uses (degrees for joints,
0-100 for grippers). Errors print `ERROR <message>`; exit code 0 = pose printed,
1 = error (no calibration, port busy, ...).

Usage:
    bmh-read-pose --left_port /dev/ttyACM1 --right_port /dev/ttyACM0 --id so_follower
"""

import argparse
import contextlib
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ReadPoseError(Exception):
    """An error the operator can act on; printed as `ERROR <message>`."""


def pose_line(left: dict[str, float], right: dict[str, float]) -> str:
    """Formats per-arm motor positions as the `POSE {json}` line with bimanual action keys."""
    pose = {f"left_{motor}.pos": round(float(value), 3) for motor, value in left.items()}
    pose.update({f"right_{motor}.pos": round(float(value), 3) for motor, value in right.items()})
    return f"POSE {json.dumps(pose)}"


def create_follower(port: str, arm_id: str, with_head: bool):
    """Instantiate (without connecting) one follower arm, like `bmh-center-arm` does."""
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    return SOFollower(SOFollowerRobotConfig(port=port, id=arm_id, with_head=with_head))


def connect_bus(arm) -> None:
    """Open the arm's bus (ping only) and make sure its calibration registers match the file.

    `arm.connect()` is avoided on purpose: its `configure()` re-enables torque.
    """
    if not arm.calibration:
        raise ReadPoseError(
            f"no calibration file found for id '{arm.id}' "
            f"(expected at {arm.calibration_fpath}). Calibrate this arm first."
        )
    bus = arm.bus
    bus.connect()
    if bus.is_calibrated:
        return
    # Rewriting Homing_Offset under torque can make a servo move, so only do it while limp.
    torque = bus.sync_read("Torque_Enable", normalize=False)
    if any(value != 0 for value in torque.values()):
        raise ReadPoseError(
            f"calibration registers of '{arm.id}' differ from the file and torque is on — "
            "run a teleoperation session or recalibrate"
        )
    logger.info("Motor calibration registers differ from the file for %s — rewriting them", arm.id)
    bus.write_calibration(arm.calibration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the present pose of both BMH-101 follower arms")
    parser.add_argument(
        "--left_port", required=True, help="USB serial port of the left arm (carries the head)"
    )
    parser.add_argument("--right_port", required=True, help="USB serial port of the right arm")
    parser.add_argument(
        "--id", default="so_follower", help="Robot id; the arms use <id>_left / <id>_right calibrations"
    )
    args = parser.parse_args()

    left = create_follower(args.left_port, f"{args.id}_left", with_head=True)
    right = create_follower(args.right_port, f"{args.id}_right", with_head=False)
    try:
        # Connect both arms before reading anything, so a half-read pose is never printed.
        for arm in (left, right):
            connect_bus(arm)
        print(
            pose_line(left.bus.sync_read("Present_Position"), right.bus.sync_read("Present_Position")),
            flush=True,
        )
    except ReadPoseError as e:
        print(f"ERROR {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Reading the pose failed")
        print(f"ERROR {' '.join(str(e).split())}", flush=True)
        sys.exit(1)
    finally:
        for arm in (left, right):
            if arm.bus.is_connected:
                with contextlib.suppress(Exception):
                    arm.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
