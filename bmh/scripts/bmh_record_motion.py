#!/usr/bin/env python
"""bmh-record-motion — record a *fixed recording* (joint trajectory only) by teleoperation.

The follower follows the network leader exactly like in a dataset recording, but only the
joint trajectory is kept — no cameras, no `LeRobotDataset` — as one ``motion.json``
(format: ``bmh/motion/file.py``).

The controller-app's Fixed recordings tab runs it on the follower Pi. Like ``bmh-record``
it takes a JSON config instead of CLI flags, decoded with ``draccus.decode`` (skips the
~5.5 s draccus CLI parse). Its ``robot`` / ``teleop`` sections are the ones the app writes
for ``bmh-record`` (here without cameras), so the teleop stack — network leader, group
lock file, follower — is the same as for a dataset recording; only the sink differs:

    {
      "robot": {"type": "bi_so_follower", "id": "so_follower",
                "left_arm_config": {"port": "/dev/ttyACM1", "with_head": true},
                "right_arm_config": {"port": "/dev/ttyACM0"}},
      "teleop": {"type": "bi_so_network_leader", "left_arm_config": {...},
                 "right_arm_config": {...}, "lock_file": "/home/pi/.cache/bmh-101/teleop-locks.json"},
      "output": "/home/pi/.cache/bmh-101/fixed-recordings/wave-1a2b/motion.json",
      "fps": 60
    }

Every tick at ``fps`` reads the leader, sends the action to the follower and stores the
action *actually sent* (after the group-lock blend and any clipping) on the 16 keys of
``MOTION_KEYS``.

Line-oriented stdout for the app:

    MOTION_READY          both sides connected; frames are recorded from the next tick
    MOTION_FRAMES <n>     frames so far, about once per second
    MOTION_SAVED <n>      the file is written (after SIGTERM / SIGINT)
    ERROR <message>       anything failed (exit code 1); nothing is saved

SIGTERM or SIGINT ends the take: the frames are saved, then the follower's torque goes off
(``robot.disconnect()``, like at the end of every session) and the leader disconnects. A
signal before ``MOTION_READY`` aborts at once without saving (exit code 1).

Usage:
    bmh-record-motion --config ~/.cache/bmh-101/fixed-recording-config.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bmh.motion.file import MOTION_KEYS, MotionFile, save_motion

# Registers the `bi_so_follower` / `bi_so_network_leader` config types for draccus.
from lerobot.robots import RobotConfig, bi_so_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig, bi_so_network_leader  # noqa: F401

logger = logging.getLogger(__name__)

# Decimals kept per value: the servos resolve ~0.09 degrees, so 3 decimals lose nothing
# and keep a 2-minute take near 1 MB.
FRAME_DECIMALS = 3


@dataclass
class RecordMotionConfig:
    """The JSON config of `bmh-record-motion` (see the module docstring)."""

    robot: RobotConfig
    teleop: TeleoperatorConfig
    # Where the `motion.json` goes (its folder is created if missing).
    output: str
    # Control rate: one leader read, one follower command and one stored frame per tick.
    fps: int = 60


class RecordMotionError(Exception):
    """An error the operator can act on; printed as `ERROR <message>`."""


def load_record_motion_config(path: Path | str) -> RecordMotionConfig:
    """Decode the JSON config file into a :class:`RecordMotionConfig`."""
    import draccus
    from draccus.utils import ParsingError

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level, got {type(data).__name__}")
    try:
        cfg = draccus.decode(RecordMotionConfig, data)
    except ParsingError as e:
        cause = f" ({e.__cause__})" if e.__cause__ else ""
        raise ValueError(f"{path}: {e}{cause}") from e
    if cfg.fps <= 0:
        raise ValueError(f"{path}: 'fps' must be positive, got {cfg.fps}")
    if not cfg.output:
        raise ValueError(f"{path}: 'output' must name the motion.json to write")
    return cfg


def record_tick(action_sent: Mapping[str, Any], keys: Sequence[str]) -> tuple[float, ...]:
    """One frame: the action sent to the follower on ``keys``, rounded to FRAME_DECIMALS.

    Raises:
        RecordMotionError: If the action lacks one of ``keys`` (the leader sent no value
            for that joint).
    """
    missing = [key for key in keys if key not in action_sent]
    if missing:
        raise RecordMotionError(f"the leader sent no value for {', '.join(missing)}")
    return tuple(round(float(action_sent[key]), FRAME_DECIMALS) for key in keys)


def missing_keys(action_features: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    """The ``keys`` the robot has no action for (the BMH-101 needs all 16, head included)."""
    return [key for key in keys if key not in action_features]


class _StopFlag:
    """SIGTERM / SIGINT handling: abort while connecting, stop gracefully once recording."""

    def __init__(self) -> None:
        self.recording = False
        self.event = threading.Event()

    def __call__(self, signum: int, frame: Any) -> None:
        if not self.recording:
            raise KeyboardInterrupt
        self.event.set()

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self)
        signal.signal(signal.SIGINT, self)


def _say(line: str) -> None:
    print(line, flush=True)


def run(cfg: RecordMotionConfig, stop: _StopFlag) -> int:
    """Connect, record until ``stop``, save. Returns the number of frames saved."""
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.cycle_timer import CycleTimer

    robot = make_robot_from_config(cfg.robot)
    missing = missing_keys(robot.action_features, MOTION_KEYS)
    if missing:
        raise RecordMotionError(
            f"the follower has no {', '.join(missing)} — a fixed recording needs both arms and the head"
        )
    teleop = make_teleoperator_from_config(cfg.teleop)
    output = Path(cfg.output).expanduser()
    frames: list[tuple[float, ...]] = []
    timer = CycleTimer(cfg.fps)
    try:
        # Leader first, so the follower is not left idle while the leader stream comes up
        # (same order as `lerobot-record`). `BiSOFollower.connect()` connects both arms in parallel.
        teleop.connect()
        robot.connect()
        recorded_at = time.time()
        stop.recording = True
        _say("MOTION_READY")
        next_report = time.monotonic() + 1.0
        while not stop.event.is_set():
            timer.tick()
            action = teleop.get_action()
            sent = robot.send_action(action)
            frames.append(record_tick(sent, MOTION_KEYS))
            now = time.monotonic()
            if now >= next_report:
                _say(f"MOTION_FRAMES {len(frames)}")
                next_report = now + 1.0
            timer.wait()
        timer.log_run_summary()
        if not frames:
            raise RecordMotionError("no frames were recorded")
        save_motion(
            output,
            MotionFile(fps=cfg.fps, keys=MOTION_KEYS, frames=tuple(frames), recorded_at=recorded_at),
        )
        _say(f"MOTION_SAVED {len(frames)}")
        return len(frames)
    finally:
        # Torque off on the follower (its default on disconnect), then let go of the leader.
        if robot.is_connected:
            with contextlib.suppress(Exception):
                robot.disconnect()
        if teleop.is_connected:
            with contextlib.suppress(Exception):
                teleop.disconnect()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: `bmh-record-motion --config <file.json>`; errors end as one `ERROR` line."""
    parser = argparse.ArgumentParser(
        prog="bmh-record-motion",
        description="Record a fixed recording (joint trajectory only) from the network leader.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the JSON config file")
    args = parser.parse_args(argv)

    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    init_logging()
    stop = _StopFlag()
    stop.install()
    try:
        register_third_party_plugins()  # parity with `bmh-record`; a few ms
        cfg = load_record_motion_config(args.config)
        run(cfg, stop)
    except KeyboardInterrupt:
        _say("ERROR stopped before the recording started")
        sys.exit(1)
    except RecordMotionError as e:
        _say(f"ERROR {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fixed recording failed")
        _say(f"ERROR {' '.join(str(e).split()) or type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
