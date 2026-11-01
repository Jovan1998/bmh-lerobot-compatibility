"""Live inference target for the BMH-101 GR00T client.

The controller-app hands ``bmh-groot-client`` its *target* — policy server host /
port / API token plus the language instruction — through a small JSON file
(``~/.cache/bmh-101/inference-control.json``) that it replaces atomically (tmp +
rename) whenever the operator applies a change. The client ``os.stat``s the file
once per control tick and re-reads it only when it changed, so a new prompt or
server is picked up within one tick and applied to the *next* inference request —
no restart, no robot reconnect. Same mechanism as the teleop group lock
(``lerobot.teleoperators.bi_so_network_leader.group_lock.LockFileWatcher``).

Every target carries a monotonically increasing ``seq`` so that re-applying the
same values after a rejected switch is a new request, and so the app can tell
"requested" from "applied". The client answers each decided ``seq`` with one
``BMH_TARGET {json}`` log line (see :func:`format_target_status`) that the app
parses back into its status; the API token is never echoed.

File contract (all keys required except ``api_token``):

.. code-block:: json

    {"seq": 3, "policy_host": "1.2.3.4", "policy_port": 5555,
     "lang_instruction": "pick up the cube", "api_token": "..."}

A target can also be *idle* — "no server, no prompt": the client closes its policy
connection, stops moving and holds position until a real target arrives. That is how
the Physical Agent tab starts the client before any skill is live. Only ``seq`` is
read from an idle document, every other key is ignored:

.. code-block:: json

    {"seq": 4, "idle": true}

An idle document may also say ``"relaxed": true`` — idle with every motor's torque
off, so the operator can move the arms by hand (the Physical Agent's "Relax"). The
torque comes back on, holding wherever the arms were left, with the next document
that is not relaxed: a plain idle ("Hold") or a real target. A non-idle document
ignores the key.

.. code-block:: json

    {"seq": 5, "idle": true, "relaxed": true}

An idle document may instead carry ``"home": {action_key: value}`` — the robot's saved
Home position (the Physical Agent's "Home position" button and a script's ``goHome()``).
The client stiffens the motors if they were relaxed, ramps every listed joint from where
it is to that pose, and then holds it like a plain idle. It answers the ``seq`` with a
second ``BMH_TARGET`` line once the pose is reached (``"home_reached": true``), never
before the first one: a status line of an older ``seq`` cannot land between them. ``home``
and ``relaxed: true`` exclude each other; a non-idle document ignores ``home``.

.. code-block:: json

    {"seq": 6, "idle": true, "home": {"left_shoulder_pan.pos": 0.5, "right_gripper.pos": 2.0}}

Idle is never expressed by deleting the file — a missing file means "keep the
current target" (see :class:`ControlFileWatcher`).

A skill trained with feedbacks also reports binary flags ("Done", custom ones). They
travel back the same way, as a line kind of their own — ``BMH_FEEDBACK {json}`` (see
:func:`format_feedback_status`) — logged only when the flags change, so the
``BMH_TARGET`` payload stays what it is.

Loop settings — how far ahead the client plays (``action_horizon``), when it asks for
the next chunk (``refetch_after``) and how fast it plays (``speed_pct`` of the recorded
fps) — come through a second file of their own (``~/.cache/bmh-101/inference-loop.json``),
written the same way (tmp + rename) and watched the same way (:class:`LoopFileWatcher`).
Its ``seq`` is independent of the target's: a settings change never re-requests a chunk
and never counts as a new target. The app never deletes it — it is also where the
settings persist between runs. All keys are required ints:

.. code-block:: json

    {"seq": 3, "action_horizon": 24, "refetch_after": 10, "speed_pct": 100}

The client answers each decided ``seq`` (and the startup read) with one
``BMH_LOOP {json}`` line (see :func:`format_loop_status`). A document that breaks a rule
(:func:`parse_loop_settings`, :func:`validate_loop_settings`) is rejected as a whole:
the line says ``"accepted": false`` with the reason and carries the values still in
effect.

stdlib only (no zmq / cv2 / hardware imports) so it is unit-testable anywhere.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Prefix of the one-line JSON status the client logs per decided target. The
# controller-app greps for it in the process output (see
# bmh-101-os/controller-app/src/lib/inference.controller.ts).
STATUS_PREFIX = "BMH_TARGET "

# Prefix of the one-line JSON snapshot the client logs whenever the policy's feedback
# flags change (same consumer). A line kind of its own, not part of `BMH_TARGET`.
FEEDBACK_STATUS_PREFIX = "BMH_FEEDBACK "

# Prefix of the one-line JSON status the client logs per decided loop-settings seq
# (same consumer). A line kind of its own: loop settings have their own file and seq.
LOOP_STATUS_PREFIX = "BMH_LOOP "

# Loop-file ranges checked at parse time. `blend_frames < action_horizon` needs the
# client's CLI value, so it is checked when the settings are applied
# (`validate_loop_settings`).
LOOP_HORIZON_MAX = 60
LOOP_SPEED_PCT_MIN = 50
LOOP_SPEED_PCT_MAX = 150
_LOOP_KEYS = ("seq", "action_horizon", "refetch_after", "speed_pct")

# Signature reported for a missing control file (never collides with a real stat).
_MISSING_SIGNATURE = (-1, -1, -1)


@dataclass(frozen=True)
class InferenceTarget:
    """One requested inference target: which server to ask, with which instruction.

    Attributes:
        seq: Monotonic request number assigned by the writer (0 for CLI-only runs).
        policy_host: GR00T policy server host.
        policy_port: GR00T policy server port.
        lang_instruction: Language instruction sent with every request.
        api_token: Per-request API token (empty = none). Hidden from ``repr``.
        idle: ``True`` for "no server, no prompt" — host / instruction / token are
            empty and the port is 0. Build one with :meth:`make_idle`.
        relaxed: Idle with every motor's torque off (only ever set on an idle target).
        home: Idle at the Home position: ``(action key, value)`` pairs, sorted by key, the
            robot ramps to and then holds. Empty for every other target; a tuple so the
            frozen dataclass stays hashable. Read it as a dict via :attr:`home_pose`.
    """

    seq: int
    policy_host: str
    policy_port: int
    lang_instruction: str
    api_token: str = field(default="", repr=False)
    idle: bool = False
    relaxed: bool = False
    home: tuple[tuple[str, float], ...] = ()

    @classmethod
    def make_idle(
        cls, seq: int, relaxed: bool = False, home: Mapping[str, float] | None = None
    ) -> InferenceTarget:
        """The idle target for ``seq``: no server to ask, nothing to do — torque off if
        ``relaxed``, moving to and holding ``home`` if given."""
        return cls(
            seq=seq,
            policy_host="",
            policy_port=0,
            lang_instruction="",
            idle=True,
            relaxed=relaxed,
            home=tuple(sorted((key, float(value)) for key, value in (home or {}).items())),
        )

    @property
    def home_pose(self) -> dict[str, float]:
        """The Home position to move to, as ``{action key: value}`` (empty when none)."""
        return dict(self.home)

    @property
    def endpoint(self) -> tuple[str, int, str]:
        """``(host, port, token)`` — what a ``PolicyClient`` is bound to.

        Equal across prompt-only changes, so the worker can tell "just swap the
        instruction" from "reconnect to another server".
        """
        return (self.policy_host, self.policy_port, self.api_token)


@dataclass(frozen=True)
class Hold:
    """Worker reply for a rejected target: hold position until a new ``seq`` arrives."""

    seq: int
    error: str


@dataclass(frozen=True)
class Idle:
    """Worker reply for an accepted idle target: stay still until a new ``seq`` arrives."""

    seq: int


def parse_target(data: object) -> InferenceTarget:
    """Validate a decoded control-file document.

    Args:
        data: Result of ``json.loads`` on the control file.

    Returns:
        The target, with host / instruction / token whitespace-stripped — or the idle
        target for ``seq`` when the document says ``"idle": true`` (of its other keys
        only ``relaxed`` and ``home`` are read).

    Raises:
        ValueError: If a required key is missing or has the wrong type / range.
            Bools are rejected where ints are expected; ``idle`` and an idle
            document's ``relaxed`` must be bools, its ``home`` a non-empty object of
            finite numbers, and ``home`` cannot be combined with ``relaxed: true``.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    seq = data.get("seq")
    if type(seq) is not int or seq < 0:
        raise ValueError(f"'seq' must be a non-negative integer, got {seq!r}")
    idle = data.get("idle", False)
    if not isinstance(idle, bool):
        raise ValueError(f"'idle' must be a boolean, got {idle!r}")
    if idle:
        relaxed = data.get("relaxed", False)
        if not isinstance(relaxed, bool):
            raise ValueError(f"'relaxed' must be a boolean, got {relaxed!r}")
        home = _parse_home(data.get("home"))
        if home and relaxed:
            raise ValueError("'home' and 'relaxed' exclude each other")
        return InferenceTarget.make_idle(seq, relaxed=relaxed, home=home)
    host = data.get("policy_host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("'policy_host' must be a non-empty string")
    port = data.get("policy_port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"'policy_port' must be an integer in [1, 65535], got {port!r}")
    lang = data.get("lang_instruction")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("'lang_instruction' must be a non-empty string")
    token = data.get("api_token", "")
    if not isinstance(token, str):
        raise ValueError(f"'api_token' must be a string, got {type(token).__name__}")
    return InferenceTarget(
        seq=seq,
        policy_host=host.strip(),
        policy_port=port,
        lang_instruction=lang.strip(),
        api_token=token.strip(),
    )


def _parse_home(value: object) -> dict[str, float] | None:
    """An idle document's optional ``home``: ``None`` when absent, else ``{key: finite number}``."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("'home' must be a non-empty object of joint positions")
    home: dict[str, float] = {}
    for key, position in value.items():
        if (
            isinstance(position, bool)
            or not isinstance(position, (int, float))
            or not math.isfinite(position)
        ):
            raise ValueError(f"'home' position for {key!r} must be a finite number, got {position!r}")
        home[key] = float(position)
    return home


@dataclass(frozen=True)
class LoopSettings:
    """How the control loop plays the chunks: horizon, refetch point and speed.

    Attributes:
        seq: Monotonic request number assigned by the app (0 for CLI-only runs). Its own
            counter, independent of the target's ``seq``.
        action_horizon: Actions kept of each arriving chunk (the rest is cut off).
        refetch_after: Actions played from a fresh chunk before the next request fires.
        speed_pct: Playback speed in percent of the recorded fps (100 = as recorded).
    """

    seq: int
    action_horizon: int
    refetch_after: int
    speed_pct: int = 100

    def period_s(self, fps: int) -> float:
        """Seconds per control tick: ``1 / (fps * speed_pct / 100)``."""
        return 100.0 / (fps * self.speed_pct)


def parse_loop_settings(data: object) -> LoopSettings:
    """Validate a decoded loop-file document.

    Args:
        data: Result of ``json.loads`` on the loop file.

    Returns:
        The settings.

    Raises:
        ValueError: If a key is missing, is not an int (bools and floats are rejected),
            or is out of range: ``seq >= 0``, ``1 <= action_horizon <= 60``,
            ``1 <= refetch_after < action_horizon``, ``50 <= speed_pct <= 150``.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    for key in _LOOP_KEYS:
        if key not in data:
            raise ValueError(f"missing {key!r}")
        if type(data[key]) is not int:
            raise ValueError(f"{key!r} must be an integer, got {data[key]!r}")
    settings = LoopSettings(
        seq=data["seq"],
        action_horizon=data["action_horizon"],
        refetch_after=data["refetch_after"],
        speed_pct=data["speed_pct"],
    )
    if settings.seq < 0:
        raise ValueError(f"'seq' must be a non-negative integer, got {settings.seq}")
    if not 1 <= settings.action_horizon <= LOOP_HORIZON_MAX:
        raise ValueError(
            f"'action_horizon' must be in [1, {LOOP_HORIZON_MAX}], got {settings.action_horizon}"
        )
    error = _check_refetch_and_speed(settings)
    if error is not None:
        raise ValueError(error)
    return settings


def _check_refetch_and_speed(settings: LoopSettings) -> str | None:
    """The rules :func:`parse_loop_settings` and :func:`validate_loop_settings` share."""
    if not 1 <= settings.refetch_after < settings.action_horizon:
        return (
            f"refetch_after must be in [1, action_horizon={settings.action_horizon}), "
            f"got {settings.refetch_after}"
        )
    if not LOOP_SPEED_PCT_MIN <= settings.speed_pct <= LOOP_SPEED_PCT_MAX:
        return f"speed_pct must be in [{LOOP_SPEED_PCT_MIN}, {LOOP_SPEED_PCT_MAX}], got {settings.speed_pct}"
    return None


def validate_loop_settings(settings: LoopSettings, blend_frames: int) -> str | None:
    """Check settings against the running client before they are used.

    One rule set for the startup settings (loop file or CLI flags) and for every live
    change: ``1 <= refetch_after < action_horizon``, ``0 <= blend_frames <
    action_horizon`` and ``50 <= speed_pct <= 150``. The blend rule needs the client's
    ``--blend_frames``, which the file parser cannot know.

    Args:
        settings: The settings to check.
        blend_frames: The client's ``--blend_frames``.

    Returns:
        ``None`` when the settings can be applied, else the reason they cannot.
    """
    error = _check_refetch_and_speed(settings)
    if error is not None:
        return error
    if blend_frames < 0:
        return f"blend_frames must be >= 0, got {blend_frames}"
    if blend_frames >= settings.action_horizon:
        return (
            f"action_horizon must be greater than blend_frames={blend_frames}, got {settings.action_horizon}"
        )
    return None


def format_loop_status(settings: LoopSettings, accepted: bool, error: str | None) -> str:
    """One-line ``BMH_LOOP {json}`` status for the app to parse.

    Args:
        settings: The settings now in effect, carrying the ``seq`` that was decided — on
            a rejection the *rejected* seq with the values that stay in effect.
        accepted: Whether the document of that ``seq`` was applied.
        error: Reason for a rejection, ``None`` when accepted.
    """
    return LOOP_STATUS_PREFIX + json.dumps(
        {
            "seq": settings.seq,
            "accepted": accepted,
            "error": error,
            "action_horizon": settings.action_horizon,
            "refetch_after": settings.refetch_after,
            "speed_pct": settings.speed_pct,
        }
    )


def smoothstep(t: float) -> float:
    """Cubic ease-in/ease-out on [0, 1]; clamps outside that range."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def ramp_pose(start: Mapping[str, float], target: Mapping[str, float], alpha: float) -> dict[str, float]:
    """Pose ``alpha`` of the way from ``start`` to ``target`` (0 = start, 1 = target).

    Covers exactly the keys of ``start``; a key ``target`` does not list keeps its start value.
    """
    return {
        key: value + (target[key] - value) * alpha if key in target else value for key, value in start.items()
    }


class _JsonFileWatcher[T]:
    """Change detector for one of the controller-app's JSON files.

    Costs one ``os.stat`` per :meth:`poll`; the file is only opened and parsed when its
    inode / mtime / size changed (i.e. once per apply). Unlike the lock-file watcher a
    *missing* file carries no meaning ("there is no unset target"): it reports ``None``
    and the caller keeps what it has; a re-created file is picked up again. Subclasses
    name the document (``_what``, for log lines) and parse it (``_parse``).
    """

    _what = "file"

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._signature: tuple[int, int, int] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def read_initial(self) -> T | None:
        """Startup read of the file the app wrote before spawning the client.

        Primes the change detector, so a following :meth:`poll` reports ``None`` unless
        the file changed after this call — the startup read is not a "change".

        Returns:
            The parsed document, or ``None`` if the file does not exist.

        Raises:
            ValueError: If the file exists but is unreadable or invalid — the app is
                the only writer, so this is a contract bug worth failing loudly on.
        """
        signature = self._stat()
        self._signature = signature
        if signature == _MISSING_SIGNATURE:
            return None
        try:
            return self._read()
        except OSError as e:
            raise ValueError(f"cannot read {self._what} {self._path}: {e}") from e

    def poll(self) -> T | None:
        """Return the new document if the file changed since the last call, else ``None``.

        A malformed or vanished file is logged once per file version and ignored
        (``None``), leaving the caller's current state untouched.
        """
        signature = self._stat()
        if signature == self._signature:
            return None
        self._signature = signature
        if signature == _MISSING_SIGNATURE:
            return None
        try:
            return self._read()
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable %s %s: %s", self._what, self._path, e)
            return None

    def _parse(self, data: object) -> T:
        raise NotImplementedError

    def _stat(self) -> tuple[int, int, int]:
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            return _MISSING_SIGNATURE
        # st_ino changes on every tmp+rename, so even same-size same-mtime rewrites register.
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _read(self) -> T:
        # json.JSONDecodeError is a ValueError, so callers see one failure type.
        return self._parse(json.loads(self._path.read_text()))


class ControlFileWatcher(_JsonFileWatcher[InferenceTarget]):
    """Change detector for the inference control file: yields :class:`InferenceTarget`."""

    _what = "control file"

    def _parse(self, data: object) -> InferenceTarget:
        return parse_target(data)


class LoopFileWatcher(_JsonFileWatcher[LoopSettings]):
    """Change detector for the loop-settings file: yields :class:`LoopSettings`.

    Only parse-time ranges are checked here; the client applies a document only after
    :func:`validate_loop_settings` against its ``--blend_frames``.
    """

    _what = "loop file"

    def _parse(self, data: object) -> LoopSettings:
        return parse_loop_settings(data)


def format_target_status(
    target: InferenceTarget, accepted: bool, error: str | None, home_reached: bool = False
) -> str:
    """One-line ``BMH_TARGET {json}`` status for the app to parse. Never includes the token.

    Args:
        target: The target that was decided on (accepted or rejected). An idle target
            reports ``"idle": true`` with an empty host / instruction and port 0, a
            relaxed one also ``"relaxed": true``, one at the Home position ``"home": true``.
        accepted: Whether the client now runs against ``target``.
        error: Reason for a rejection, ``None`` when accepted.
        home_reached: ``True`` on the second line of a Home target, logged once the
            robot arrived at the pose (same ``seq``).
    """
    return STATUS_PREFIX + json.dumps(
        {
            "seq": target.seq,
            "accepted": accepted,
            "error": error,
            "idle": target.idle,
            "policy_host": target.policy_host,
            "policy_port": target.policy_port,
            "lang_instruction": target.lang_instruction,
            "relaxed": target.relaxed,
            "home": bool(target.home),
            "home_reached": home_reached,
        }
    )


def format_feedback_status(seq: int, flags: Mapping[str, bool], values: Mapping[str, float]) -> str:
    """One-line ``BMH_FEEDBACK {json}`` snapshot of the policy's feedback flags.

    Compact JSON, keys in this order:
    ``BMH_FEEDBACK {"seq":4,"flags":{"done":true},"values":{"done":0.93}}``.

    Args:
        seq: ``seq`` of the target the flags belong to; the app shows them only while
            that target is the applied one.
        flags: Full snapshot, ``{feedback key: on?}`` (see
            ``bmh/groot_client/feedback.py::FeedbackTracker``). Empty = cleared.
        values: Raw 0..1 value behind each flag at the moment of the change, for the
            log reader. Rounded to 3 decimals; non-finite values are left out so the
            line is always valid JSON.
    """
    return FEEDBACK_STATUS_PREFIX + json.dumps(
        {
            "seq": seq,
            "flags": dict(flags),
            "values": {k: round(float(v), 3) for k, v in values.items() if math.isfinite(v)},
        },
        separators=(",", ":"),
    )
