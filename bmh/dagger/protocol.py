"""Wire contract between the controller-app and `bmh-dagger`.

**Commands** arrive on stdin, one JSON object per line. Every command carries an integer
``id`` and a ``cmd``; the process answers each with one ``BMH_DAGGER_ACK`` line:

.. code-block:: json

    {"id": 1, "cmd": "policy", "policy_host": "1.2.3.4", "policy_port": 5555,
     "api_token": "...", "lang_instruction": "put the cube in the bowl"}
    {"id": 2, "cmd": "run"}
    {"id": 3, "cmd": "pause"}
    {"id": 4, "cmd": "step", "cursor": 45}
    {"id": 5, "cmd": "home", "pose": {"left_shoulder_pan.pos": 0.0}}
    {"id": 6, "cmd": "align", "task": "put the cube in the bowl", "auto_start": true,
     "assist": false, "assist_torque_pct": 10}
    {"id": 7, "cmd": "record"}
    {"id": 8, "cmd": "cancel"}
    {"id": 9, "cmd": "save"}
    {"id": 10, "cmd": "discard"}
    {"id": 11, "cmd": "finish"}
    {"id": 12, "cmd": "assist", "assist": true, "assist_torque_pct": 15}

``assist`` switches the leader assist (the leader arms pull lightly toward the pose the
robot holds) on or off, or changes its torque, during a take-over; ``align`` sets it at the
start of one.

stdin rather than a polled file (the policy client's control file) because these are
events, not a desired state: two of them inside one tick must both arrive, in order. The
API token never touches the disk, and stdin EOF (the app died) ends the session instead
of leaving a policy running unsupervised.

**Status** goes out on stdout:

.. code-block:: text

    DAGGER_READY
    BMH_DAGGER_ACK {"id": 6, "ok": true}
    BMH_DAGGER_ACK {"id": 7, "ok": false, "error": "not aligning"}
    BMH_DAGGER {"phase": "aligning", "policy": "ready", "history": 412, "cursor": 0,
                "episodes": 3, "hold": {...}, "align": {...}, "take": {"frames": 0}}
    ERROR <message>

These lines are printed, not logged: the process's log goes to stderr, and a log line of
level ERROR would otherwise look like the final ``ERROR`` line.

stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

import json
import math
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import IO, Literal, TypeGuard

READY_LINE = "DAGGER_READY"
STATUS_PREFIX = "BMH_DAGGER "
ACK_PREFIX = "BMH_DAGGER_ACK "
ERROR_PREFIX = "ERROR "

Phase = Literal[
    "connecting", "paused", "running", "stepping", "homing", "aligning", "recording", "saving", "finishing"
]
PolicyState = Literal["none", "connecting", "ready", "error"]
AssistState = Literal["off", "on", "unavailable"]

# Range of `assist_torque_pct`: percent of a servo's full torque, so 10 is a `Torque_Limit`
# of 100. 20 % is what the leader arms were bench-tested with (2026-11-12); the leader host
# caps the limit again by itself (`so_leader_host --assist_max_torque_limit`).
ASSIST_TORQUE_PCT_MIN = 1
ASSIST_TORQUE_PCT_MAX = 20
ASSIST_TORQUE_PCT_DEFAULT = 10


@dataclass(frozen=True)
class Policy:
    """Use this policy server and instruction (probed by the inference worker)."""

    id: int
    policy_host: str
    policy_port: int
    lang_instruction: str
    api_token: str = field(default="", repr=False)


@dataclass(frozen=True)
class Run:
    """Let the policy drive, from the pose at the cursor."""

    id: int


@dataclass(frozen=True)
class Pause:
    """Stop where the robot is and hold."""

    id: int


@dataclass(frozen=True)
class Step:
    """Walk the robot to history frame ``cursor`` (frames back from the newest, 0 = newest)."""

    id: int
    cursor: int


@dataclass(frozen=True)
class Home:
    """Ramp to the robot's saved Home position."""

    id: int
    pose: Mapping[str, float]


@dataclass(frozen=True)
class Align:
    """Begin a take-over: hold, and compare the leader with the held pose."""

    id: int
    task: str
    auto_start: bool = True
    assist: bool = False
    assist_torque_pct: int = ASSIST_TORQUE_PCT_DEFAULT


@dataclass(frozen=True)
class Assist:
    """Switch the leader assist on or off, or change its torque, while aligning."""

    id: int
    assist: bool
    assist_torque_pct: int = ASSIST_TORQUE_PCT_DEFAULT


@dataclass(frozen=True)
class Record:
    """Start recording now, aligned or not."""

    id: int


@dataclass(frozen=True)
class Cancel:
    """Abandon the take-over before recording started."""

    id: int


@dataclass(frozen=True)
class Save:
    """End the take and keep it as an episode."""

    id: int


@dataclass(frozen=True)
class Discard:
    """End the take and throw it away."""

    id: int


@dataclass(frozen=True)
class Finish:
    """Finalize the dataset and exit."""

    id: int


Command = Policy | Run | Pause | Step | Home | Align | Assist | Record | Cancel | Save | Discard | Finish


@dataclass(frozen=True)
class Eof:
    """stdin closed: the app is gone."""


@dataclass(frozen=True)
class Invalid:
    """A line that is not a command. ``id`` is set when the line still named one, so it can be acked."""

    error: str
    id: int | None = None


_PLAIN: dict[str, type[Run | Pause | Record | Cancel | Save | Discard | Finish]] = {
    "run": Run,
    "pause": Pause,
    "record": Record,
    "cancel": Cancel,
    "save": Save,
    "discard": Discard,
    "finish": Finish,
}


def _is_int(value: object) -> TypeGuard[int]:
    return type(value) is int


def _command_id(data: Mapping[str, object]) -> int:
    value = data.get("id")
    if not _is_int(value) or value < 0:
        raise ValueError(f"'id' must be a non-negative integer, got {value!r}")
    return value


def _flag(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be a boolean, got {value!r}")
    return value


def _text(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value.strip()


def _assist_torque_pct(data: Mapping[str, object]) -> int:
    value = data.get("assist_torque_pct", ASSIST_TORQUE_PCT_DEFAULT)
    if not _is_int(value) or not ASSIST_TORQUE_PCT_MIN <= value <= ASSIST_TORQUE_PCT_MAX:
        raise ValueError(
            f"'assist_torque_pct' must be an integer in [{ASSIST_TORQUE_PCT_MIN}, "
            f"{ASSIST_TORQUE_PCT_MAX}], got {value!r}"
        )
    return value


def _pose(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("'pose' must be a non-empty object of joint positions")
    pose: dict[str, float] = {}
    for key, position in value.items():
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
        ):
            raise ValueError(f"'pose' position for {key!r} must be a finite number, got {position!r}")
        pose[str(key)] = float(position)
    return pose


def parse_command(line: str) -> Command:
    """Decode one stdin line.

    Raises:
        ValueError: With a message naming the problem: not a JSON object, a missing or
            non-integer ``id``, an unknown ``cmd``, or a field of the command with the
            wrong type or range (``cursor`` not an int >= 0, ``pose`` not a non-empty map
            of finite numbers, ``assist_torque_pct`` outside 1..20, an empty ``task``,
            a missing ``policy_host`` / ``lang_instruction``, a port outside 1..65535).
    """
    try:
        data = json.loads(line)
    except ValueError as e:
        raise ValueError(f"not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    command_id = _command_id(data)
    cmd = data.get("cmd")
    if isinstance(cmd, str) and cmd in _PLAIN:
        return _PLAIN[cmd](command_id)
    if cmd == "policy":
        port = data.get("policy_port")
        if not _is_int(port) or not 1 <= port <= 65535:
            raise ValueError(f"'policy_port' must be an integer in [1, 65535], got {port!r}")
        token = data.get("api_token", "")
        if not isinstance(token, str):
            raise ValueError(f"'api_token' must be a string, got {type(token).__name__}")
        return Policy(
            id=command_id,
            policy_host=_text(data, "policy_host"),
            policy_port=port,
            lang_instruction=_text(data, "lang_instruction"),
            api_token=token.strip(),
        )
    if cmd == "step":
        cursor = data.get("cursor")
        if not _is_int(cursor) or cursor < 0:
            raise ValueError(f"'cursor' must be a non-negative integer, got {cursor!r}")
        return Step(command_id, cursor)
    if cmd == "home":
        return Home(command_id, _pose(data.get("pose")))
    if cmd == "align":
        return Align(
            id=command_id,
            task=_text(data, "task"),
            auto_start=_flag(data, "auto_start", True),
            assist=_flag(data, "assist", False),
            assist_torque_pct=_assist_torque_pct(data),
        )
    if cmd == "assist":
        return Assist(
            id=command_id,
            assist=_flag(data, "assist", False),
            assist_torque_pct=_assist_torque_pct(data),
        )
    raise ValueError(f"unknown cmd {cmd!r}")


def parse_line(line: str) -> Command | Invalid:
    """`parse_command`, with a rejected line turned into an `Invalid` that keeps its ``id``."""
    try:
        return parse_command(line)
    except ValueError as e:
        command_id: int | None = None
        try:
            data = json.loads(line)
            if isinstance(data, dict) and _is_int(data.get("id")):
                command_id = data["id"]
        except ValueError:
            pass
        return Invalid(error=str(e), id=command_id)


def format_ack(command_id: int, ok: bool, error: str | None = None) -> str:
    """``BMH_DAGGER_ACK {"id": 7, "ok": false, "error": "not aligning"}`` (no ``error`` key when ok)."""
    payload: dict[str, object] = {"id": command_id, "ok": ok}
    if error is not None:
        payload["error"] = error
    return ACK_PREFIX + json.dumps(payload)


def _rounded(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 2) for key, value in values.items()}


def format_align(
    diff: Mapping[str, float], aligned: bool, leader_age_ms: float, assist: AssistState
) -> dict[str, object]:
    """The ``align`` object of a status line: per-joint ``leader - hold``, 2 decimals."""
    return {
        "diff": _rounded(diff),
        "aligned": aligned,
        "leader_age_ms": int(round(leader_age_ms)),
        "assist": assist,
    }


def format_status(
    phase: Phase,
    policy: PolicyState,
    history: int,
    cursor: int,
    episodes: int,
    *,
    hold: Mapping[str, float] | None = None,
    align: Mapping[str, object] | None = None,
    take_frames: int | None = None,
) -> str:
    """One ``BMH_DAGGER {json}`` line; keys in this fixed order, optional ones left out.

    Args:
        phase: Session phase.
        policy: State of the policy connection.
        history: Frames in the rewind history.
        cursor: Frames back from the newest history frame the robot is at.
        episodes: Episodes saved to the staging dataset so far.
        hold: The pose the robot holds (while paused / aligning), 2 decimals.
        align: `format_align` output (while aligning).
        take_frames: Frames in the running take (while aligning / recording / saving).
    """
    payload: dict[str, object] = {
        "phase": phase,
        "policy": policy,
        "history": history,
        "cursor": cursor,
        "episodes": episodes,
    }
    if hold is not None:
        payload["hold"] = _rounded(hold)
    if align is not None:
        payload["align"] = dict(align)
    if take_frames is not None:
        payload["take"] = {"frames": take_frames}
    return STATUS_PREFIX + json.dumps(payload)


class StdinReader:
    """Reads command lines on a daemon thread, so the control loop never blocks on stdin."""

    def __init__(self, stream: IO[str]):
        """Wrap ``stream`` (stdin); nothing is read before `start`."""
        self._stream = stream
        self._queue: queue.Queue[Command | Eof | Invalid] = queue.Queue()
        self._thread = threading.Thread(target=self._read, name="bmh-dagger-stdin", daemon=True)

    def start(self) -> None:
        """Begin reading."""
        self._thread.start()

    def _read(self) -> None:
        try:
            for line in self._stream:
                if line.strip():
                    self._queue.put(parse_line(line))
        finally:
            # Also on a read error: without the app there is nobody to supervise the robot.
            self._queue.put(Eof())

    def drain(self) -> list[Command | Eof | Invalid]:
        """Everything that arrived since the last call, in order (never blocks)."""
        items: list[Command | Eof | Invalid] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                return items
