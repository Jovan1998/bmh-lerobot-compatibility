#!/usr/bin/env python
r"""bmh-replay-motion — play a fixed recording on the BMH-101 follower, then hold its last pose.

The controller-app's Fixed recordings tab runs it when nothing else holds the serial ports
(while the Physical Agent runs, ``bmh-policy-client`` plays recordings instead — same
``bmh/motion/player.py`` plan, same motion). One-shot, like ``bmh-center-arm``:

1. Open both buses only (no ``robot.connect()``: its ``configure()`` would turn the torque
   on with whatever goal the servos still hold), check / write the calibration like
   ``bmh-read-pose``, pin every goal to the present position and only then enable torque —
   so nothing jumps.
2. Plan the replay from the present pose (``goto-start``: ramp to the first kept frame over
   ``--ramp_s``, then play; ``blend-in``: play at once, fading the offset out over
   ``--blend_s``) and send ``plan.pose_at(t)`` at ``--fps`` until the last frame.
3. Hold the final pose (torque on) until a newline arrives on stdin, stdin closes, or
   SIGTERM / SIGINT — then torque off and exit 0.

SIGUSR1 during playback stops the motion where it is and holds there (step 3).

Line-oriented stdout for the app:

    REPLAY_PHASE ramping     moving to the recording's first frame (goto-start only)
    REPLAY_PHASE playing     playing the recording
    REPLAY_DONE              the last frame was played; holding the final pose
    REPLAY_PHASE holding     stopped by SIGUSR1; holding where the arms are
    ERROR <message>          anything failed (exit code 1); torque is turned off

Usage:
    bmh-replay-motion --left_port /dev/ttyACM1 --right_port /dev/ttyACM0 --id so_follower \\
        --file ~/.cache/bmh-101/fixed-recordings/wave-1a2b/motion.json --start 12 --end 610 \\
        --mode blend-in
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any

from bmh.motion.file import load_motion
from bmh.motion.player import DEFAULT_BLEND_S, DEFAULT_RAMP_S, MODES, Phase, ReplayPlan, plan_replay
from bmh.scripts.bmh_read_pose import ReadPoseError, connect_bus, create_follower

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Retries per torque / position transfer when stiffening and releasing (as `bus.disconnect` uses).
_RETRIES = 5
_PREFIXES = ("left_", "right_")


def split_pose(pose: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """``{action key: value}`` → ``{"left_": {motor: value}, "right_": {...}}`` for the buses.

    Keys without an arm prefix or ``.pos`` suffix are ignored.
    """
    per_arm: dict[str, dict[str, float]] = {prefix: {} for prefix in _PREFIXES}
    for key, value in pose.items():
        for prefix in _PREFIXES:
            if key.startswith(prefix) and key.endswith(".pos"):
                per_arm[prefix][key.removeprefix(prefix).removesuffix(".pos")] = float(value)
    return per_arm


def as_action(prefix: str, positions: Mapping[str, Any]) -> dict[str, float]:
    """One bus's ``{motor: position}`` as action keys (``<prefix><motor>.pos``)."""
    return {f"{prefix}{motor}.pos": float(value) for motor, value in positions.items()}


def _say(line: str) -> None:
    print(line, flush=True)


def play(plan: ReplayPlan, buses: Mapping[str, Any], fps: int, hold_now: threading.Event) -> bool:
    """Send ``plan.pose_at(t)`` at ``fps`` until done, or until ``hold_now`` is set.

    Returns:
        ``True`` when the last frame was played, ``False`` when stopped early.
    """
    from lerobot.utils.cycle_timer import CycleTimer

    timer = CycleTimer(fps)
    # Taken on the first tick, so that tick sends exactly the plan's start.
    t0: float | None = None
    phase: Phase | None = None
    try:
        while True:
            timer.tick()
            if hold_now.is_set():
                return False
            now = time.perf_counter()
            if t0 is None:
                t0 = now
            t = now - t0
            now_phase = plan.phase_at(t)
            for prefix, goals in split_pose(plan.pose_at(t)).items():
                if goals:
                    buses[prefix].sync_write("Goal_Position", goals)
            if now_phase == "done":
                return True
            if now_phase != phase:
                phase = now_phase
                _say(f"REPLAY_PHASE {phase}")
            timer.wait()
    finally:
        timer.log_run_summary()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: play, hold, release on stdin / SIGTERM; errors end as one `ERROR` line."""
    parser = argparse.ArgumentParser(
        prog="bmh-replay-motion",
        description="Play a fixed recording on both BMH-101 follower arms, then hold the last pose",
    )
    parser.add_argument(
        "--left_port", required=True, help="USB serial port of the left arm (carries the head)"
    )
    parser.add_argument("--right_port", required=True, help="USB serial port of the right arm")
    parser.add_argument(
        "--id", default="so_follower", help="Robot id; the arms use <id>_left / <id>_right calibrations"
    )
    parser.add_argument("--file", required=True, help="The recording's motion.json")
    parser.add_argument("--start", type=int, default=0, help="First frame to play (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End of the frames to play (exclusive; default: all)"
    )
    parser.add_argument(
        "--mode", choices=MODES, default="goto-start", help="How the arms get onto the recording"
    )
    parser.add_argument(
        "--ramp_s", type=float, default=DEFAULT_RAMP_S, help="goto-start: ramp to the first frame"
    )
    parser.add_argument("--blend_s", type=float, default=DEFAULT_BLEND_S, help="blend-in: offset fade-out")
    parser.add_argument("--fps", type=int, default=60, help="Goal-position writes per second")
    args = parser.parse_args(argv)

    hold_now = threading.Event()

    def on_usr1(signum: int, frame: Any) -> None:
        hold_now.set()

    def on_term(signum: int, frame: Any) -> None:
        # Let go like ^C: torque off in the `finally` below (the default handler skips it).
        raise KeyboardInterrupt

    previous_handlers = {
        signal.SIGUSR1: signal.signal(signal.SIGUSR1, on_usr1),
        signal.SIGTERM: signal.signal(signal.SIGTERM, on_term),
    }

    left = create_follower(args.left_port, f"{args.id}_left", with_head=True)
    right = create_follower(args.right_port, f"{args.id}_right", with_head=False)
    arms = (left, right)
    try:
        if args.fps <= 0:
            raise ReadPoseError(f"--fps must be positive, got {args.fps}")
        motion = load_motion(args.file)
        end = len(motion.frames) if args.end is None else args.end

        for arm in arms:
            connect_bus(arm)
        # Pin each goal to where the arm is before the torque comes on, so nothing jumps.
        present: dict[str, float] = {}
        for prefix, arm in zip(_PREFIXES, arms, strict=True):
            positions = arm.bus.sync_read("Present_Position", num_retry=_RETRIES)
            arm.bus.sync_write("Goal_Position", positions, num_retry=_RETRIES)
            arm.bus.enable_torque(num_retry=_RETRIES)
            present.update(as_action(prefix, positions))

        plan = plan_replay(
            present, motion, args.start, end, args.mode, ramp_s=args.ramp_s, blend_s=args.blend_s
        )
        if plan.dropped_keys:
            logger.warning(
                "Ignoring recording keys the robot does not have: %s", ", ".join(plan.dropped_keys)
            )
        logger.info(
            "Playing %s (%s, frames %d-%d of %d, %.1f s)",
            args.file,
            args.mode,
            args.start,
            end,
            len(motion.frames),
            plan.duration_s,
        )

        buses = {prefix: arm.bus for prefix, arm in zip(_PREFIXES, arms, strict=True)}
        if play(plan, buses, args.fps, hold_now):
            _say("REPLAY_DONE")
        else:
            _say("REPLAY_PHASE holding")

        # Hold under torque until the operator releases (newline / EOF on stdin, SIGTERM, ^C).
        with contextlib.suppress(EOFError, KeyboardInterrupt):
            sys.stdin.readline()
    except KeyboardInterrupt:
        logger.info("Released")
    except (ReadPoseError, ValueError, OSError) as e:
        _say(f"ERROR {' '.join(str(e).split())}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Replay failed")
        _say(f"ERROR {' '.join(str(e).split()) or type(e).__name__}")
        sys.exit(1)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for arm in arms:
            if arm.bus.is_connected:
                try:
                    arm.bus.disconnect(disable_torque=True)
                except Exception:
                    logger.warning("Failed to release %s cleanly", arm.id, exc_info=True)


if __name__ == "__main__":
    main()
