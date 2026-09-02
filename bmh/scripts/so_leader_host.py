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
"""

import argparse
import contextlib
import json
import logging
import time

import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_push_socket(context: zmq.Context, port: int) -> zmq.Socket:
    sock = context.socket(zmq.PUSH)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.bind(f"tcp://*:{port}")
    logger.info(f"ZMQ PUSH bound on tcp://*:{port}")
    return sock


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
    args = parser.parse_args()

    if not args.left_port and not args.right_port:
        parser.error("At least one of --left_port or --right_port must be specified")

    zmq_context = zmq.Context()
    leaders = []

    try:
        if args.left_port:
            left_leader = create_leader(args.left_port, args.left_id, with_head=args.left_with_head)
            left_socket = create_push_socket(zmq_context, args.left_zmq_port)
            leaders.append(("left", left_leader, left_socket))
            logger.info(f"Left leader arm on {args.left_port} → ZMQ :{args.left_zmq_port}")

        if args.right_port:
            right_leader = create_leader(args.right_port, args.right_id)
            right_socket = create_push_socket(zmq_context, args.right_zmq_port)
            leaders.append(("right", right_leader, right_socket))
            logger.info(f"Right leader arm on {args.right_port} → ZMQ :{args.right_zmq_port}")

        logger.info(f"Streaming at {args.fps} Hz. Press Ctrl+C to stop.")
        loop_dt = 1.0 / args.fps

        while True:
            loop_start = time.perf_counter()

            for name, leader, socket in leaders:
                try:
                    action = leader.get_action()
                except ConnectionError as e:
                    logger.warning(f"{name} arm read error, skipping frame: {e}")
                    continue
                # zmq.Again: no receiver connected yet, drop frame
                with contextlib.suppress(zmq.Again):
                    socket.send_string(json.dumps(action), zmq.NOBLOCK)

            elapsed = time.perf_counter() - loop_start
            sleep_time = loop_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_dt = time.perf_counter() - loop_start
            logger.debug(f"Loop: {actual_dt * 1e3:.1f}ms ({1 / actual_dt:.0f} Hz)")

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        for name, leader, socket in leaders:
            try:
                leader.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {name} leader: {e}")
            socket.close()
        zmq_context.term()
        logger.info("Leader host stopped.")


if __name__ == "__main__":
    main()
