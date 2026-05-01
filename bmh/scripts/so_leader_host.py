#!/usr/bin/env python
"""
Leader Host — runs on Pi A (leader Pi).

Reads two SO-101 leader arms via USB and streams their joint positions
over ZMQ PUSH sockets to the follower Pi.

Usage:
    python so_leader_host.py \
        --left_port /dev/ttyACM0 \
        --right_port /dev/ttyACM1 \
        --left_zmq_port 5555 \
        --right_zmq_port 5557 \
        --fps 60

Single arm mode:
    python so_leader_host.py \
        --left_port /dev/ttyACM0 \
        --left_zmq_port 5555 \
        --fps 60
"""

import argparse
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


def create_leader(port: str, arm_id: str):
    """Create and connect an SOLeader arm."""
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig

    config = SOLeaderTeleopConfig(port=port, id=arm_id)
    leader = SOLeader(config)
    leader.connect()
    return leader


def main():
    parser = argparse.ArgumentParser(description="SO-101 Leader Host — streams arm positions over ZMQ")
    parser.add_argument("--left_port", type=str, default=None, help="USB serial port for left leader arm")
    parser.add_argument("--right_port", type=str, default=None, help="USB serial port for right leader arm")
    parser.add_argument("--left_zmq_port", type=int, default=5555, help="ZMQ port for left arm stream")
    parser.add_argument("--right_zmq_port", type=int, default=5557, help="ZMQ port for right arm stream")
    parser.add_argument("--fps", type=int, default=60, help="Target loop frequency in Hz")
    parser.add_argument("--left_id", type=str, default="bmh_leader_left", help="Calibration ID for left arm")
    parser.add_argument("--right_id", type=str, default="bmh_leader_right", help="Calibration ID for right arm")
    args = parser.parse_args()

    if not args.left_port and not args.right_port:
        parser.error("At least one of --left_port or --right_port must be specified")

    zmq_context = zmq.Context()
    leaders = []
    sockets = []

    try:
        if args.left_port:
            left_leader = create_leader(args.left_port, args.left_id)
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
                try:
                    socket.send_string(json.dumps(action), zmq.NOBLOCK)
                except zmq.Again:
                    pass  # no receiver connected yet, drop frame

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
