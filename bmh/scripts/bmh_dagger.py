#!/usr/bin/env python
"""bmh-dagger — DAgger recording on the BMH-101: a policy runs, the operator takes over.

One long-running process on the follower Pi owns the robot for a whole session. It
connects the follower, the cameras, the leader stream and the dataset writer once, and
switches between policy, hold, rewind and teleoperation inside one loop, so a switch
costs one tick. (`bmh-policy-client` and `bmh-record` each need the serial ports and the
cameras, and ~10 s to start, so they cannot be alternated.)

Only the operator's correction is recorded: each take-over becomes one episode of a v3
staging dataset that has exactly the features of a `bmh-record` staging dataset. The
connection itself is not: the follower follows the leader for `takeover_settle_s` before
the first frame is written, so the bounce of the connection (the blend, the leader assist
letting go, the operator taking the arms' weight) stays out of the episode.

Session phases:

    connecting  not connected yet                          -> paused (prints DAGGER_READY)
    paused      holds history[cursor]                      -> run, step, home, align, finish
    running     policy chunks (look-ahead, smoothing)      -> pause, align; policy error -> paused
    stepping    one history frame per tick to the cursor   -> arrival -> paused; pause, run, step, home, align
    homing      smoothstep ramp to the Home pose           -> arrival -> paused; pause
    aligning    holds; compares the leader with the hold   -> aligned 0.4 s (auto_start) or record
                                                              -> settling; cancel -> paused
    settling    follows the leader (blended in), writes    -> 3 s later -> recording; cancel -> paused
                nothing yet
    recording   follows the leader, writes                 -> save / discard -> saving
    saving      servos keep their last goal (loop blocked) -> paused
    finishing   dataset.finalize(), disconnect             -> exit 0

The controller-app drives it with JSON lines on stdin and reads `BMH_DAGGER` /
`BMH_DAGGER_ACK` lines from stdout (see `bmh/dagger/protocol.py`). A command that is not
valid in the current phase is refused, never queued. `finish`, stdin EOF (the app died),
SIGTERM / SIGINT and any error all end the session through the same path: an unsaved take
is discarded, the dataset is finalized, robot and leader are disconnected. An error then
prints `ERROR <message>` and exits 1, so saved episodes survive a hardware fault.

Reused as they are from the policy client: the adapter, the policy connect, the inference
worker (which logs the `BMH_TARGET` lines), look-ahead and smoothing, and the loop
settings file (`--loop_file` of `bmh-policy-client`, answered with the same `BMH_LOOP`
lines), so corrections are collected under the motion shaping the policy is deployed with.

Leader assist: with `assist` on (in `align`, or the `assist` command while aligning) and a
leader stream that has the feedback channel (`feedback_port_zmq` on both arms of the
`bi_so_network_leader` config), every aligning tick sends the hold pose of the arm joints
and grippers that are not locked to the leader host, which pulls the leader arms toward it
with a low torque. Leaving `aligning` (the arms connect, cancel, finish, an error) releases
the leader at once; the leader host also lets go by itself 0.3 s after the messages stop.
Without the channel the status reports `"assist": "unavailable"`.

Rewind: every action sent while the robot is being moved (policy, teleop) goes into a ring
buffer, one frame per 30 Hz tick. `step` carries an absolute cursor and the loop walks
toward it by one frame per tick, so the robot replays its own motion and lost or repeated
key events cannot make it jump. Running or recording from a rewound cursor drops the
frames after it.

The JSON config mirrors `bmh-record`'s (`robot`, `teleop` and `dataset` are the same
blocks):

    {
      "robot": {"type": "bi_so_follower", ...},
      "teleop": {"type": "bi_so_network_leader", ...},
      "dataset": {"repo_id": "local/my-set", "root": "/path/to/staging", ...},
      "control_fps": 60,
      "loop_file": "/home/pi/.cache/bmh-101/inference-loop.json",
      "jpeg_quality": 80
    }

Usage:
    bmh-dagger --config ~/.cache/bmh-101/dagger-config.json
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Protocol

from bmh.dagger.history import ActionHistory
from bmh.dagger.protocol import (
    ERROR_PREFIX,
    READY_LINE,
    Align,
    Assist,
    AssistState,
    Cancel,
    Command,
    Discard,
    Eof,
    Finish,
    Home,
    Invalid,
    Pause,
    Phase,
    Policy,
    Record,
    Run,
    Save,
    StdinReader,
    Step,
    format_ack,
    format_align,
    format_status,
)
from bmh.dagger.takeover import AlignGate, AlignState, TakeoverBlend, alignment
from bmh.motion.player import smoothstep
from bmh.policy_client.control import (
    LoopFileWatcher,
    LoopSettings,
    format_loop_status,
    ramp_pose,
    validate_loop_settings,
)

# Importing the camera, robot and teleoperator configs registers their `type` names, which
# `draccus.decode` needs (the network leaders register with `lerobot.teleoperators` itself).
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.robots import RobotConfig, bi_so_follower, so_follower  # noqa: F401
from lerobot.teleoperators import TeleoperatorConfig

logger = logging.getLogger(__name__)

# The policy's control rate, and the rate of the rewind history: one frame per tick.
POLICY_FPS = 30
# Status lines per second: normally, and while the operator aligns the leader.
STATUS_HZ = 2.0
ALIGN_STATUS_HZ = 10.0
# A leader message older than this means the stream is dead; a take must not start on it.
LEADER_STALE_S = 0.5
# `assist_torque_pct` is percent of full torque; the servos' `Torque_Limit` counts in 0.1 %.
TORQUE_LIMIT_PER_PCT = 10


@dataclass
class DaggerConfig:
    """JSON configuration of `bmh-dagger` (decoded with `draccus.decode`, like `bmh-record`)."""

    robot: RobotConfig
    teleop: TeleoperatorConfig
    dataset: DatasetRecordConfig
    # Rate the arms are commanded at while the operator steers. A multiple of `dataset.fps`:
    # a frame is recorded on every `control_fps // dataset.fps`-th command, as in `bmh-record`.
    control_fps: int = 60
    # The controller-app's loop-settings file (see bmh/policy_client/control.py), polled
    # every tick. Missing file or `None`: the policy client's default settings.
    loop_file: str | None = None
    jpeg_quality: int = 80
    # Seconds of sent actions kept for rewinding.
    history_s: float = 60.0
    # The leader counts as aligned when every joint is within these of the hold pose
    # (degrees; 0..100 for the grippers) ...
    align_arm_tol_deg: float = 4.0
    align_gripper_tol: float = 5.0
    # ... for this long without a gap; then, with `auto_start`, the arms connect by themselves.
    align_hold_s: float = 0.4
    # Shortest fade of the leader / follower offset at the start of a take.
    takeover_blend_s: float = 1.0
    # Seconds the follower follows the leader before the take's first frame is written
    # (`settling`); 0 writes from the connection on.
    takeover_settle_s: float = 3.0
    home_ramp_s: float = 2.5
    # Policy server timeouts, as for `bmh-policy-client`.
    probe_timeout_ms: int = 5000
    timeout_ms: int = 15000

    def __post_init__(self) -> None:
        """Reject values the loop cannot run with."""
        if self.control_fps <= 0 or self.control_fps % self.dataset.fps != 0:
            raise ValueError(
                f"control_fps ({self.control_fps}) must be a positive multiple of "
                f"dataset.fps ({self.dataset.fps})."
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}.")
        for name in ("history_s", "home_ramp_s", "takeover_blend_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")
        if not self.takeover_settle_s >= 0:
            raise ValueError(f"takeover_settle_s must be >= 0, got {self.takeover_settle_s}.")

    @property
    def control_multiplier(self) -> int:
        """Commands per recorded frame during a take."""
        return self.control_fps // self.dataset.fps


def load_dagger_config(path: Path | str) -> DaggerConfig:
    """Decode a JSON file into a `DaggerConfig`.

    Raises:
        ValueError: If the file is not a JSON object or does not decode.
    """
    import draccus
    from draccus.utils import ParsingError

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level, got {type(data).__name__}")
    try:
        return draccus.decode(DaggerConfig, data)
    except ParsingError as e:
        # draccus hides the dataclass' own error behind a generic message; surface it.
        cause = f" ({e.__cause__})" if e.__cause__ else ""
        raise ValueError(f"{path}: {e}{cause}") from e


class _Runner(Protocol):
    """What the session needs of `bmh.dagger.policy_runner.PolicyRunner`."""

    @property
    def policy_state(self) -> str: ...
    @property
    def settings(self) -> LoopSettings: ...
    def period_s(self) -> float: ...
    def set_target(self, host: str, port: int, lang_instruction: str, api_token: str) -> Any: ...
    def apply_settings(self, settings: LoopSettings) -> None: ...
    def resume(self, pose: dict[str, float]) -> None: ...
    def suspend(self) -> None: ...
    def tick(self, tick_start: float) -> dict[str, float] | None: ...
    def fire(self, get_observation: Callable[[], dict[str, Any]]) -> bool: ...


class _Recorder(Protocol):
    """What the session needs of `bmh.dagger.recorder.TakeRecorder`."""

    frames: int

    @property
    def episodes(self) -> int: ...
    def add(self, observation: dict[str, Any], action: dict[str, float], task: str) -> None: ...
    def save(self) -> bool: ...
    def discard(self) -> None: ...


@dataclass
class _HomeRamp:
    start: dict[str, float]
    target: dict[str, float]
    t0: float | None = None


class DaggerSession:
    """The session's state machine: one `tick` per control period.

    Owns no hardware set-up or tear-down (see `run_session`); it is handed a connected
    robot and leader, the take recorder and the policy runner, and is driven by the
    commands `read_commands` returns.
    """

    def __init__(
        self,
        robot: Any,
        teleop: Any,
        recorder: _Recorder,
        runner: _Runner,
        cfg: DaggerConfig,
        *,
        read_commands: Callable[[], list[Command | Eof | Invalid]],
        emit: Callable[[str], None],
        loop_watcher: LoopFileWatcher | None = None,
        stop_requested: Callable[[], bool] = lambda: False,
    ):
        """Wire the session up; nothing moves before `start`.

        Args:
            robot: Connected follower.
            teleop: Connected leader; its ``last_message_age_s`` (when it has one) guards
                the start of a take against a dead stream.
            recorder: Writes the takes.
            runner: Plays the policy.
            cfg: The session's configuration.
            read_commands: Returns the commands that arrived since the last call.
            emit: Writes one protocol line.
            loop_watcher: Live loop settings, ``None`` for fixed ones.
            stop_requested: Polled every tick; ``True`` finishes the session.
        """
        self._robot = robot
        self._teleop = teleop
        self._recorder = recorder
        self._runner = runner
        self._cfg = cfg
        self._read_commands = read_commands
        self._emit = emit
        self._loop_watcher = loop_watcher
        self._stop_requested = stop_requested

        self.phase: Phase = "connecting"
        self._history = ActionHistory(max(1, round(cfg.history_s * POLICY_FPS)))
        self._cursor = 0
        self._step_target = 0
        self._home: _HomeRamp | None = None
        # The last action sent to the robot (what it holds between ticks).
        self._last_sent: dict[str, float] = {}

        self._task = ""
        self._auto_start = True
        self._assist: AssistState = "off"
        self._assist_torque_limit = 0
        # A goal went to the leader since the last release.
        self._assisting = False
        self._gate = AlignGate(cfg.align_hold_s)
        self._align: AlignState | None = None
        self._leader: dict[str, float] | None = None
        self._leader_age_s = 0.0

        self._blend: TakeoverBlend | None = None
        self._take_t0: float | None = None
        self._ticks_in_frame = 0
        self._frame_start = 0.0
        self._max_take_frames = max(1, round(cfg.dataset.episode_time_s * cfg.dataset.fps))
        self._save_pending: bool | None = None

        self._finish = False
        self._reported_phase: Phase | None = None
        self._status_at = 0.0

    # ------------------------------------------------------------------ public

    @property
    def cursor(self) -> int:
        """Frames back from the newest history frame the robot is at."""
        return self._cursor

    @property
    def history(self) -> ActionHistory:
        """The rewind history."""
        return self._history

    @property
    def hold(self) -> dict[str, float]:
        """The pose the robot holds while nothing moves it: the history frame at the cursor."""
        return self._history.at(self._cursor)

    def start(self, pose: dict[str, float], now: float) -> None:
        """Both sides are connected and the arms are at ``pose``: hold it and report ready."""
        self._history.clear()
        self._history.push(pose)
        self._cursor = 0
        self._last_sent = dict(pose)
        self.phase = "paused"
        self._emit(READY_LINE)
        self._emit_status(now, force=True)

    def period_s(self) -> float:
        """Seconds until the next tick is due, by phase."""
        if self.phase == "running":
            return self._runner.period_s()
        if self.phase in ("aligning", "settling", "recording"):
            return 1.0 / self._cfg.control_fps
        return 1.0 / POLICY_FPS

    def run(
        self,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Tick until the session is finished."""
        while True:
            tick_start = clock()
            if not self.tick(tick_start):
                return
            remaining = self.period_s() - (clock() - tick_start)
            if remaining > 0:
                sleep(remaining)

    def tick(self, now: float) -> bool:
        """One control tick.

        Returns:
            ``False`` once the session is finished (`finish`, stdin EOF or a stop signal);
            the caller then finalizes.
        """
        for item in self._read_commands():
            self._handle(item, now)
        if self._stop_requested() and not self._finish:
            logger.info("Stop signal received — finishing the session.")
            self._finish = True
        if self._finish:
            self._release_assist()
            if self.phase == "saving":
                # Save was pressed just before the end: the take must not be lost to it.
                self._finish_take()
            self.phase = "finishing"
            self._emit_status(now, force=True)
            return False
        self._poll_loop_settings()

        if self.phase == "saving":
            # The tick that ended the take reports `saving`; the next one blocks in the save.
            if self._reported_phase == "saving":
                self._finish_take()
            else:
                self._send(self.hold)
        elif self.phase == "running":
            self._tick_running(now)
        elif self.phase == "stepping":
            self._tick_stepping()
        elif self.phase == "homing":
            self._tick_homing(now)
        elif self.phase == "aligning":
            self._tick_aligning(now)
        elif self.phase in ("settling", "recording"):
            self._tick_take(now)
        else:
            self._send(self.hold)
        if self.phase in ("paused", "stepping", "homing"):
            # A target set while the policy does not drive is still probed (its chunk is
            # dropped), so the app sees `ready` or the error before Run.
            self._runner.tick(now)
            self._runner.fire(self._robot.get_observation)
        if self.phase != "aligning" or self._assist != "on":
            # The same tick the arms connect (or the take-over ends): the leader goes limp.
            self._release_assist()
        self._emit_status(now)
        return True

    # ---------------------------------------------------------------- commands

    def _handle(self, item: Command | Eof | Invalid, now: float) -> None:
        if isinstance(item, Eof):
            logger.warning("stdin closed (the controller-app is gone) — finishing the session.")
            self._finish = True
            return
        if isinstance(item, Invalid):
            logger.warning("Ignoring an invalid command line: %s", item.error)
            if item.id is not None:
                self._emit(format_ack(item.id, False, item.error))
            return
        error = self._apply(item, now)
        if error is not None:
            logger.warning("command #%d (%s) refused: %s", item.id, type(item).__name__.lower(), error)
        self._emit(format_ack(item.id, error is None, error))

    def _apply(self, cmd: Command, now: float) -> str | None:
        """Carry out one command. Returns why it was refused, or ``None``."""
        phase = self.phase
        if isinstance(cmd, Finish):
            self._finish = True
            return None
        if isinstance(cmd, Policy):
            if phase in ("aligning", "settling", "recording", "saving"):
                return "not during a take-over"
            self._runner.set_target(cmd.policy_host, cmd.policy_port, cmd.lang_instruction, cmd.api_token)
            return None
        if isinstance(cmd, Run):
            if phase not in ("paused", "stepping"):
                return "not paused"
            if self._runner.policy_state == "none":
                return "no policy set"
            self._settle_at_cursor()
            self._runner.resume(self.hold)
            self.phase = "running"
            return None
        if isinstance(cmd, Pause):
            if phase == "running":
                self._runner.suspend()
            elif phase == "homing":
                self._end_home()
            elif phase != "stepping":
                return "nothing to pause"
            self.phase = "paused"
            return None
        if isinstance(cmd, Step):
            if phase not in ("paused", "stepping"):
                return "not paused"
            self._step_target = min(cmd.cursor, len(self._history) - 1)
            self.phase = "paused" if self._step_target == self._cursor else "stepping"
            return None
        if isinstance(cmd, Home):
            if phase not in ("paused", "stepping"):
                return "not paused"
            target = {key: value for key, value in cmd.pose.items() if key in self._robot.action_features}
            if not target:
                return "the Home position names no joint of this robot"
            self._home = _HomeRamp(start=self.hold, target=target)
            self.phase = "homing"
            return None
        if isinstance(cmd, Align):
            if phase not in ("paused", "running", "stepping"):
                return "not paused or running"
            if phase == "running":
                self._runner.suspend()
            self._task = cmd.task
            self._auto_start = cmd.auto_start
            self._set_assist(cmd.assist, cmd.assist_torque_pct)
            self._gate.reset()
            self._align = None
            self._leader = None
            self.phase = "aligning"
            return None
        if isinstance(cmd, Assist):
            if phase != "aligning":
                return "not aligning"
            self._set_assist(cmd.assist, cmd.assist_torque_pct)
            return None
        if isinstance(cmd, Record):
            if phase != "aligning":
                return "not aligning"
            leader = self._read_leader()
            if self._leader_age_s > LEADER_STALE_S:
                return "the leader stream is stale"
            self._begin_take(leader)
            return None
        if isinstance(cmd, Cancel):
            if phase == "settling":
                # Nothing was written yet; the robot stays where the leader took it.
                self._stop_following()
            elif phase != "aligning":
                return "nothing to cancel"
            self.phase = "paused"
            return None
        if isinstance(cmd, (Save, Discard)):
            if phase != "recording":
                return "not recording"
            self._end_take(save=isinstance(cmd, Save))
            return None
        return f"unhandled command {type(cmd).__name__}"

    # ------------------------------------------------------------------ phases

    def _send(self, action: dict[str, float]) -> None:
        self._robot.send_action(action)
        self._last_sent = action

    def _settle_at_cursor(self) -> None:
        """Make the frame at the cursor the newest: what follows starts from here."""
        self._history.truncate(self._cursor)
        self._cursor = 0

    def _tick_running(self, now: float) -> None:
        action = self._runner.tick(now)
        if self._runner.policy_state == "error":
            # Rejected server or failed request (reported by the worker's BMH_TARGET line).
            self._runner.suspend()
            self.phase = "paused"
            self._send(self.hold)
            return
        if action is None:
            self._send(self.hold)
        else:
            self._send(action)
            self._history.push(action)
        self._runner.fire(self._robot.get_observation)

    def _tick_stepping(self) -> None:
        if self._cursor != self._step_target:
            self._cursor += 1 if self._step_target > self._cursor else -1
        self._send(self.hold)
        if self._cursor == self._step_target:
            self.phase = "paused"

    def _tick_homing(self, now: float) -> None:
        ramp = self._home
        if ramp is None:
            self.phase = "paused"
            return
        if ramp.t0 is None:
            ramp.t0 = now
        alpha = smoothstep(min(1.0, (now - ramp.t0) / self._cfg.home_ramp_s))
        self._send(ramp_pose(ramp.start, ramp.target, alpha))
        if alpha >= 1.0:
            logger.info("Home position reached")
            self._end_home()
            self.phase = "paused"

    def _end_home(self) -> None:
        """The rollout before a Home move is over: the history restarts at the pose reached."""
        self._home = None
        self._history.clear()
        self._history.push(self._last_sent)
        self._cursor = 0

    def _read_leader(self) -> dict[str, float]:
        leader = self._teleop.get_action()
        self._leader = leader
        self._leader_age_s = float(getattr(self._teleop, "last_message_age_s", 0.0))
        return leader

    def _set_assist(self, on: bool, torque_pct: int) -> None:
        if not on:
            self._assist = "off"
        elif getattr(self._teleop, "has_feedback", False):
            self._assist = "on"
        else:
            # The leader stream was started without the follower-to-leader channel.
            self._assist = "unavailable"
        self._assist_torque_limit = torque_pct * TORQUE_LIMIT_PER_PCT

    def _pull_leader(self, hold: dict[str, float]) -> None:
        """Send the leader the hold pose of every arm joint whose group follows the leader."""
        locked: frozenset[str] = getattr(self._teleop, "locked_keys", frozenset())
        goal = {key: hold[key] for key in self._teleop.feedback_features if key in hold and key not in locked}
        self._teleop.send_feedback(goal, self._assist_torque_limit)
        self._assisting = True

    def _release_assist(self) -> None:
        if self._assisting:
            self._assisting = False
            self._teleop.release_feedback()

    def _tick_aligning(self, now: float) -> None:
        hold = self.hold
        leader = self._read_leader()
        self._align = alignment(leader, hold, self._cfg.align_arm_tol_deg, self._cfg.align_gripper_tol)
        self._send(hold)
        if self._assist == "on":
            self._pull_leader(hold)
        fresh = self._leader_age_s <= LEADER_STALE_S
        if self._gate.update(self._align.aligned and fresh, now) and self._auto_start:
            self._begin_take(leader)

    def _begin_take(self, leader: dict[str, float]) -> None:
        self._settle_at_cursor()
        self._blend = TakeoverBlend(self.hold, leader, self._cfg.takeover_blend_s)
        self._take_t0 = None
        self._ticks_in_frame = 0
        self._frame_start = 0.0
        self._recorder.frames = 0
        self.phase = "settling"
        logger.info(
            "arms connected (blend %.2f s); recording in %.1f s",
            self._blend.duration_s,
            self._cfg.takeover_settle_s,
        )

    def _tick_take(self, now: float) -> None:
        """Settling or recording: the follower follows the leader, blended in from the hold pose."""
        blend = self._blend
        if blend is None:
            raise RuntimeError("following the leader without a take")
        if self._take_t0 is None:
            self._take_t0 = now
        connected_s = now - self._take_t0
        # 1 µs of slack: tick times are sums of float periods.
        if self.phase == "settling" and connected_s >= self._cfg.takeover_settle_s - 1e-6:
            self.phase = "recording"
            self._ticks_in_frame = 0  # the take's first frame is this tick's
            logger.info('recording: "%s"', self._task)
        writing = self.phase == "recording"
        # A frame every `control_multiplier` ticks, or earlier once the current frame has
        # used up its 1/fps (as `record_loop`: recorded frames come first). While settling the
        # same ticks only feed the rewind history.
        record_tick = (
            self._ticks_in_frame % self._cfg.control_multiplier == 0
            or now - self._frame_start >= 1.0 / self._cfg.dataset.fps
        )
        observation: dict[str, Any] = {}
        if record_tick:
            self._frame_start = now
            self._ticks_in_frame = 0
            if writing:
                observation = self._robot.get_observation()
        action = blend.apply(self._read_leader(), connected_s)
        self._send(action)
        if record_tick:
            if writing:
                # The action recorded is the action sent, as in `record_loop`.
                self._recorder.add(observation, action, self._task)
            self._history.push(action)
        self._ticks_in_frame += 1
        if writing and self._recorder.frames >= self._max_take_frames:
            logger.info("take reached dataset.episode_time_s — saving it")
            self._end_take(save=True)

    def _stop_following(self) -> None:
        # The frame the robot stopped on becomes the newest, so paused holds exactly it.
        if self._last_sent and self._history.at(0) != self._last_sent:
            self._history.push(self._last_sent)
        self._cursor = 0
        self._blend = None

    def _end_take(self, save: bool) -> None:
        self._stop_following()
        self._save_pending = save
        self.phase = "saving"

    def _finish_take(self) -> None:
        """Save or drop the take. Blocks while videos encode; the servos keep their last goal."""
        if self._save_pending:
            if self._recorder.save():
                logger.info("take saved as episode %d", self._recorder.episodes - 1)
        else:
            self._recorder.discard()
            logger.info("take discarded")
        self._save_pending = None
        self.phase = "paused"

    # ---------------------------------------------------------------- settings

    def _poll_loop_settings(self) -> None:
        """Pick up live loop settings, answered with a `BMH_LOOP` line like the policy client."""
        if self._loop_watcher is None:
            return
        new = self._loop_watcher.poll()
        current = self._runner.settings
        if new is None or new.seq == current.seq:
            return
        reason = validate_loop_settings(new)
        if reason is not None:
            logger.error("loop settings #%d rejected: %s", new.seq, reason)
            logger.info(format_loop_status(replace(current, seq=new.seq), False, reason))
            return
        self._runner.apply_settings(new)
        logger.info(format_loop_status(new, True, None))

    # ------------------------------------------------------------------ status

    def _emit_status(self, now: float, force: bool = False) -> None:
        interval = 1.0 / (ALIGN_STATUS_HZ if self.phase in ("aligning", "settling") else STATUS_HZ)
        # 1 ms of slack: tick times are sums of float periods and land just short of the interval.
        if not force and self.phase == self._reported_phase and now - self._status_at < interval - 1e-3:
            return
        self._reported_phase = self.phase
        self._status_at = now
        align = None
        if self.phase == "aligning":
            state = self._align
            align = format_align(
                state.diff if state else {},
                bool(state and state.aligned),
                self._leader_age_s * 1000.0,
                self._assist,
            )
        record_in_s = None
        if self.phase == "settling":
            connected_s = 0.0 if self._take_t0 is None else now - self._take_t0
            record_in_s = max(0.0, self._cfg.takeover_settle_s - connected_s)
        self._emit(
            format_status(
                self.phase,
                self._runner.policy_state,  # type: ignore[arg-type]
                len(self._history),
                self._cursor,
                self._recorder.episodes,
                hold=self.hold if self.phase in ("paused", "aligning") else None,
                align=align,
                take_frames=self._recorder.frames
                if self.phase in ("aligning", "settling", "recording", "saving")
                else None,
                record_in_s=record_in_s,
            )
        )


def _connect_both(robot: Any, teleop: Any) -> None:
    """Connect the leader stream and the robot side by side; raise the first failure."""
    errors: list[BaseException] = []

    def connect_teleop() -> None:
        try:
            teleop.connect()
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            errors.append(e)

    thread = threading.Thread(target=connect_teleop, name="bmh-dagger-teleop-connect", daemon=True)
    thread.start()
    try:
        robot.connect()
    finally:
        thread.join()
    if errors:
        raise errors[0]


def _emit_line(line: str) -> None:
    print(line, flush=True)


def run_session(
    cfg: DaggerConfig,
    robot: Any,
    teleop: Any,
    *,
    stream: IO[str],
    emit: Callable[[str], None] = _emit_line,
    stop_requested: Callable[[], bool] = lambda: False,
    policy_connect: Callable[[Any], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run one session on a not-yet-connected robot and leader, through to the clean-up.

    Args:
        cfg: The decoded configuration.
        robot: A `bi_so_follower` (its arms' buses are read for the start pose).
        teleop: The leader teleoperator.
        stream: Where the command lines come from (stdin).
        emit: Writes one protocol line (stdout).
        stop_requested: Polled every tick; ``True`` finishes the session (signals).
        policy_connect: Opens a policy client for a target; default is the policy
            client's `_connect_policy`. Tests pass a fake.
        clock: Monotonic clock in seconds.
        sleep: Waits out the rest of a tick.

    Returns:
        The process exit code: 0 after a finished session, 1 after an error (reported
        with an ``ERROR`` line once everything is cleaned up).
    """
    # Heavy imports (torch via lerobot.datasets, cv2) only here, after argument errors.
    from bmh.dagger.policy_runner import PolicyRunner
    from bmh.dagger.recorder import TakeRecorder, open_dataset
    from bmh.scripts.bmh_policy_client import (
        BiSoBimanualAdapter,
        BmhInferenceConfig,
        _connect_policy,
        _initial_loop_settings,
        _present_pose,
    )
    from lerobot.datasets import VideoEncodingManager

    error: BaseException | None = None
    runner: PolicyRunner | None = None
    try:
        dataset = open_dataset(cfg.dataset, robot)
        with VideoEncodingManager(dataset):
            recorder = TakeRecorder(dataset)
            # Errors are caught inside the context: it must exit cleanly, so that it
            # finalizes the episodes saved so far instead of cancelling their videos.
            try:
                _connect_both(robot, teleop)
                logger.info(
                    "Robot and leader connected. Recording at %d fps, arm commands at %d Hz",
                    cfg.dataset.fps,
                    cfg.control_fps,
                )
                adapter = BiSoBimanualAdapter(
                    None, jpeg_quality=cfg.jpeg_quality, joint_names=list(robot.action_features.keys())
                )

                def connect(target: Any) -> Any:
                    if policy_connect is not None:
                        return policy_connect(target)
                    return _connect_policy(target, adapter, cfg.probe_timeout_ms, cfg.timeout_ms)

                loop_watcher = LoopFileWatcher(Path(cfg.loop_file).expanduser()) if cfg.loop_file else None
                settings = _initial_loop_settings(loop_watcher, BmhInferenceConfig(robot=cfg.robot))
                logger.info(format_loop_status(settings, True, None))
                runner = PolicyRunner(adapter, connect, settings, POLICY_FPS)

                reader = StdinReader(stream)
                reader.start()
                session = DaggerSession(
                    robot,
                    teleop,
                    recorder,
                    runner,
                    cfg,
                    read_commands=reader.drain,
                    emit=emit,
                    loop_watcher=loop_watcher,
                    stop_requested=stop_requested,
                )
                session.start(_present_pose(robot), clock())
                session.run(clock, sleep)
            except (Exception, SystemExit) as e:
                logger.exception("session failed")
                error = e
            finally:
                recorder.discard()
        logger.info("Dataset finalized: %d episode(s) saved.", dataset.num_episodes)
    except (Exception, SystemExit) as e:
        logger.exception("session failed")
        error = error or e
    finally:
        if runner is not None:
            runner.close()
        for name, device in (("robot", robot), ("leader", teleop)):
            try:
                if device.is_connected:
                    device.disconnect()
            except Exception as e:
                logger.error("Error disconnecting the %s: %s", name, e)
    if error is not None:
        emit(f"{ERROR_PREFIX}{type(error).__name__}: {error}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    """Entry point of `bmh-dagger`."""
    parser = argparse.ArgumentParser(
        prog="bmh-dagger",
        description="DAgger recording session on the BMH-101, driven by JSON commands on stdin.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the DaggerConfig JSON file")
    args = parser.parse_args(argv)

    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.utils import init_logging

    init_logging()
    register_third_party_plugins()
    try:
        cfg = load_dagger_config(args.config)
    except (OSError, ValueError) as e:
        _emit_line(f"{ERROR_PREFIX}{e}")
        sys.exit(1)

    # SIGTERM / SIGINT finish the session through the normal path (finalize, disconnect).
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    sys.exit(run_session(cfg, robot, teleop, stream=sys.stdin, stop_requested=stop.is_set))


if __name__ == "__main__":
    main()
