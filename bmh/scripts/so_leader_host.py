#!/usr/bin/env python
"""
Leader Host — runs on Pi A (leader Pi).

Reads two SO-101 leader arms via USB and streams their joint positions
over ZMQ PUSH sockets to the follower Pi.

BMH-101: the head servos (head_pan/head_tilt, IDs 8/9) ride the LEFT arm's
serial bus, so the left leader is created with with_head=True by default
(disable with --no-left_with_head). The follower side must be configured
with the matching `with_head` value: if the leader streams head keys the
follower doesn't expect, the follower raises a KeyError; if the follower
expects head keys the leader doesn't send, the follower head silently
holds position.

Usage:
    python so_leader_host.py \
        --left_port /dev/ttyACM0 \
        --right_port /dev/ttyACM1 \
        --left_zmq_port 5555 \
        --right_zmq_port 5557 \
        --left_with_head \
        --fps 60

Single arm mode:
    python so_leader_host.py \
        --left_port /dev/ttyACM0 \
        --left_zmq_port 5555 \
        --fps 60

Leader assist (DAgger take-over): with `--left_feedback_port` / `--right_feedback_port` the
host also binds one PULL socket per arm, on which the follower's `bmh-dagger` sends the pose
the robot holds while the operator lines the leader up with it. The arm joints and the
gripper (never the head) are then driven toward it with a low torque limit, capped at
`--assist_max_torque_limit`, and released by themselves 0.3 s after the messages stop
(see `bmh/dagger/leader_assist.py`). Position streaming is unchanged.

    python so_leader_host.py ... --left_feedback_port 5556 --right_feedback_port 5558
"""

import argparse
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import zmq

from bmh.dagger.leader_assist import (
    DEFAULT_MAX_TORQUE_LIMIT,
    TORQUE_LIMIT_MAX,
    AssistGoal,
    AssistRelease,
    LeaderAssist,
    parse_message,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Never driven by the leader assist: the follower holds its head during a take-over.
HEAD_MOTORS = ("head_pan", "head_tilt")


@dataclass
class _Arm:
    name: str
    leader: Any
    socket: zmq.Socket
    feedback: zmq.Socket | None = None
    assist: LeaderAssist | None = None


def create_push_socket(context: zmq.Context, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ PUSH bound on tcp://*:{port}")
    return sock


def create_pull_socket(context: zmq.Context, port: int) -> zmq.Socket:
    """Bind the leader-assist socket of one arm.

    No CONFLATE here (the sender conflates): a conflating receiver drops a message it has
    not read yet when the sender disconnects, and that is the follower's last `release`.
    `receive_feedback` drains to the latest instead.
    """
    sock = context.socket(zmq.PULL)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ PULL (leader assist) bound on tcp://*:{port}")
    return sock


def receive_feedback(sock: zmq.Socket, joints: list[str]) -> AssistGoal | AssistRelease | None:
    """The latest leader-assist message on ``sock``, without blocking (``None``: nothing new)."""
    raw = None
    while True:
        try:
            raw = sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            break
    return None if raw is None else parse_message(raw, joints)


def create_assist(leader, name: str, max_torque_limit: int) -> LeaderAssist:
    """The leader assist for one arm: every motor of its bus but the head's."""
    joints = [motor for motor in leader.bus.motors if motor not in HEAD_MOTORS]
    return LeaderAssist(leader.bus, joints, max_torque_limit, name=f"{name} leader")


def create_leader(port: str, arm_id: str, with_head: bool = False):
    """Create and connect an SOLeader arm."""
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    config = SOLeaderTeleopConfig(port=port, id=arm_id, with_head=with_head)
    leader = SOLeader(config)

    if not leader.calibration:
        raise RuntimeError(
            f"No calibration file found for id '{arm_id}' "
            f"(expected at {leader.calibration_fpath}). Calibrate this arm first."
        )

    # connect(calibrate=False) never triggers the interactive calibration flow —
    # this host runs headless (no stdin), so a prompt would crash with EOFError.
    leader.connect(calibrate=False)
    if not leader.bus.is_calibrated:
        # Motor registers drifted from the calibration file (e.g. replaced servo):
        # rewrite the file values to the motors, exactly like the calibrate flow does.
        logger.info(f"{arm_id}: motor calibration registers differ from the file — rewriting them")
        leader.bus.write_calibration(leader.calibration)
    return leader


def main():
    parser = argparse.ArgumentParser(description="SO-101 Leader Host — streams arm positions over ZMQ")
    parser.add_argument("--left_port", type=str, default=None, help="USB serial port for left leader arm")
    parser.add_argument("--right_port", type=str, default=None, help="USB serial port for right leader arm")
    parser.add_argument("--left_zmq_port", type=int, default=5555, help="ZMQ port for left arm stream")
    parser.add_argument("--right_zmq_port", type=int, default=5557, help="ZMQ port for right arm stream")
    parser.add_argument("--fps", type=int, default=60, help="Target loop frequency in Hz")
    parser.add_argument("--left_id", type=str, default="so_leader_left", help="Calibration ID for left arm")
    parser.add_argument(
        "--right_id", type=str, default="so_leader_right", help="Calibration ID for right arm"
    )
    parser.add_argument(
        "--left_with_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the left arm's bus carries the head servos (head_pan/head_tilt, IDs 8/9)",
    )
    parser.add_argument(
        "--left_feedback_port",
        type=int,
        default=None,
        help="ZMQ port to receive leader-assist goals for the left arm on (no assist when unset)",
    )
    parser.add_argument(
        "--right_feedback_port",
        type=int,
        default=None,
        help="ZMQ port to receive leader-assist goals for the right arm on (no assist when unset)",
    )
    parser.add_argument(
        "--assist_max_torque_limit",
        type=int,
        default=DEFAULT_MAX_TORQUE_LIMIT,
        help=f"Highest Torque_Limit (1..{TORQUE_LIMIT_MAX}) the leader assist drives a motor with; "
        "higher requests are clamped to it",
    )
    args = parser.parse_args()

    if not args.left_port and not args.right_port:
        parser.error("At least one of --left_port or --right_port must be specified")
    if not 1 <= args.assist_max_torque_limit <= TORQUE_LIMIT_MAX:
        parser.error(f"--assist_max_torque_limit must be in [1, {TORQUE_LIMIT_MAX}]")

    zmq_context = zmq.Context()
    arms: list[_Arm] = []

    try:
        sides = (
            (
                "left",
                args.left_port,
                args.left_id,
                args.left_with_head,
                args.left_zmq_port,
                args.left_feedback_port,
            ),
            ("right", args.right_port, args.right_id, False, args.right_zmq_port, args.right_feedback_port),
        )
        for name, port, arm_id, with_head, zmq_port, feedback_port in sides:
            if not port:
                continue
            leader = create_leader(port, arm_id, with_head=with_head)
            arm = _Arm(name, leader, create_push_socket(zmq_context, zmq_port))
            arms.append(arm)
            logger.info(f"{name.capitalize()} leader arm on {port} → ZMQ :{zmq_port}")
            if feedback_port is not None:
                arm.feedback = create_pull_socket(zmq_context, feedback_port)
                arm.assist = create_assist(leader, name, args.assist_max_torque_limit)
                logger.info(
                    f"{name.capitalize()} leader assist ← ZMQ :{feedback_port} "
                    f"(joints {','.join(arm.assist.joints)}, max Torque_Limit {args.assist_max_torque_limit})"
                )

        logger.info(f"Streaming at {args.fps} Hz. Press Ctrl+C to stop.")
        loop_dt = 1.0 / args.fps

        while True:
            loop_start = time.perf_counter()

            for arm in arms:
                try:
                    action = arm.leader.get_action()
                except ConnectionError as e:
                    logger.warning(f"{arm.name} arm read error, skipping frame: {e}")
                    action = None
                if action is not None:
                    # zmq.Again: no receiver connected yet, drop frame
                    with contextlib.suppress(zmq.Again):
                        arm.socket.send_string(json.dumps(action), zmq.NOBLOCK)
                if arm.assist is not None and arm.feedback is not None:
                    # After the read: a goal written now is acted on while the next one is due.
                    message = receive_feedback(arm.feedback, arm.assist.joints)
                    arm.assist.update(time.perf_counter(), action, message)

            elapsed = time.perf_counter() - loop_start
            sleep_time = loop_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_dt = time.perf_counter() - loop_start
            logger.debug(f"Loop: {actual_dt * 1e3:.1f}ms ({1 / actual_dt:.0f} Hz)")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        for arm in arms:
            if arm.assist is not None:
                # Before the disconnect, while the bus is still open: never leave a leader stiff.
                arm.assist.release()
            try:
                arm.leader.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {arm.name} leader: {e}")
            arm.socket.close()
            if arm.feedback is not None:
                arm.feedback.close()
        zmq_context.term()
        logger.info("Leader host stopped.")


if __name__ == "__main__":
    main()
