"""Per-group freeze and unlock-blend for bimanual network teleoperation (BMH-101).

Pure, hardware-free helpers used by ``BiSONetworkLeader.get_action()``:

- ``ActionGroupLock`` overrides the keys of a *locked* group (left arm, right arm,
  head) with the values they had when the lock engaged, and on unlock eases them
  back to the live leader values over a fixed duration (smoothstep), so the
  follower never jumps. A held group can also carry an additive *offset* (the head
  jitter: a small random pan / tilt the controller-app draws per recorded episode);
  the group then holds ``held + offset`` and eases to each new offset.
- ``LockFileWatcher`` turns a small JSON state file written by the controller-app
  into lock-state updates. It costs one ``os.stat`` per poll; the file is only
  opened and parsed when its inode / mtime / size changed (i.e. once per toggle).

Because the lock is applied inside ``get_action()``, ``lerobot-record`` stores the
held / blended pose as the dataset ``action`` - exactly what the follower was
commanded.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..so_network_leader.so_network_leader import ARM_MOTOR_NAMES, HEAD_MOTOR_NAMES

logger = logging.getLogger(__name__)

LOCK_GROUPS: tuple[str, ...] = ("left", "right", "head")

# Axes of the lock file's optional ``head_jitter`` object, in degrees.
HEAD_JITTER_AXES: tuple[str, ...] = ("pan", "tilt")
# Largest per-axis head jitter accepted from the lock file; larger values are clamped.
MAX_HEAD_JITTER_DEG = 15.0

# Signature reported for a missing lock file (never collides with a real stat).
_MISSING_SIGNATURE = (-1, -1, -1)

# An offset ease advances by tick time, capped at this many seconds per tick. A pause in
# apply() calls (lerobot-record saving an episode between the reset phase and the next
# episode) then resumes the ease where it stopped instead of finishing it in one jump.
_MAX_OFFSET_TICK_S = 0.1


def bimanual_lock_groups(left_with_head: bool) -> dict[str, list[str]]:
    """Action-dict keys per lock group for ``BiSONetworkLeader``.

    Groups are explicit key sets rather than ``left_`` prefix matches because the
    head servos ride the left arm's stream (``left_head_pan.pos`` / ``left_head_tilt.pos``)
    yet must lock independently of the left arm.
    """
    return {
        "left": [f"left_{motor}.pos" for motor in ARM_MOTOR_NAMES],
        "right": [f"right_{motor}.pos" for motor in ARM_MOTOR_NAMES],
        "head": [f"left_{motor}.pos" for motor in HEAD_MOTOR_NAMES] if left_with_head else [],
    }


def head_jitter_offsets(head_jitter: dict[str, float]) -> dict[str, float]:
    """Action-key offsets for ``BiSONetworkLeader`` from a per-axis head jitter (degrees)."""
    return {f"left_head_{axis}.pos": degrees for axis, degrees in head_jitter.items()}


def smoothstep(t: float) -> float:
    """Cubic ease-in/ease-out on [0, 1]; clamps outside that range."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


@dataclass
class _GroupState:
    keys: list[str]
    mode: str = "track"  # "track" | "hold" | "blend"
    held: dict[str, float] = field(default_factory=dict)
    blend_start: float = 0.0
    # Offset added to `held` while in "hold": eases offset_from -> offset_to as
    # offset_progress goes 0 -> 1.
    offset_from: dict[str, float] = field(default_factory=dict)
    offset_to: dict[str, float] = field(default_factory=dict)
    offset_progress: float = 1.0


class ActionGroupLock:
    """Freezes groups of action keys and blends them back on release.

    ``apply()`` reconciles the *desired* lock state (``set_locks``) with each group's
    current mode:

    - ``track`` -> ``hold``: capture the group's values from the previous output, so a
      lock issued mid-blend holds the blended pose rather than the leader pose.
    - ``hold`` -> ``blend``: start easing from the last commanded values toward the live
      leader values at ``now``; after ``blend_s`` seconds the group is back in ``track``.

    While a group is held, its keys also carry the desired offsets (``set_offsets``): the
    output is ``held + offset``, and a changed offset is eased in over ``blend_s`` seconds
    of tick time. The held values themselves never change, so offsets cannot accumulate.

    ``now`` is injected (seconds, any monotonic clock) so the class is fully testable.
    """

    def __init__(self, groups: dict[str, list[str]], blend_s: float = 0.8):
        if blend_s < 0:
            raise ValueError(f"blend_s must be >= 0, got {blend_s}")
        self._groups = {name: _GroupState(keys=list(keys)) for name, keys in groups.items()}
        self._blend_s = blend_s
        self._desired = dict.fromkeys(groups, False)
        self._offsets: dict[str, float] = {}
        self._last_output: dict[str, float] = {}
        self._last_now: float | None = None

    @property
    def locks(self) -> dict[str, bool]:
        """Desired lock state per group."""
        return dict(self._desired)

    @property
    def modes(self) -> dict[str, str]:
        """Current mode per group (``track`` / ``hold`` / ``blend``)."""
        return {name: state.mode for name, state in self._groups.items()}

    def set_locks(self, locks: dict[str, bool]) -> None:
        """Update the desired state; groups not mentioned keep their current setting."""
        for name, locked in locks.items():
            if name in self._desired:
                self._desired[name] = bool(locked)
            else:
                logger.warning("Ignoring unknown lock group %r (known: %s)", name, list(self._desired))

    def set_offsets(self, offsets: dict[str, float]) -> None:
        """Replace the desired additive offset per action key; keys not mentioned mean 0.

        An offset only acts while its key's group is held, and is ignored otherwise.
        """
        self._offsets = dict(offsets)

    def apply(self, action: dict[str, float], now: float) -> dict[str, float]:
        """Return a new action dict with locked groups held and unlocking groups blended."""
        out = dict(action)
        tick_s = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), _MAX_OFFSET_TICK_S)
        self._last_now = now
        for name, group in self._groups.items():
            if self._desired[name]:
                if group.mode != "hold":
                    # Engage: hold what we last commanded (the blended value if mid-blend),
                    # falling back to the live leader value on the very first tick.
                    group.held = {
                        key: self._last_output.get(key, action[key]) for key in group.keys if key in action
                    }
                    group.mode = "hold"
                    # Start without an offset, so engaging never moves the group.
                    group.offset_from = dict.fromkeys(group.held, 0.0)
                    group.offset_to = dict(group.offset_from)
                    group.offset_progress = 1.0
                offset = self._ease_offset(group, tick_s)
                out.update({key: held + offset[key] for key, held in group.held.items()})
                continue

            if group.mode == "hold":
                # Release from the pose last commanded (held + offset), so the blend starts there.
                group.held = {key: self._last_output.get(key, held) for key, held in group.held.items()}
                group.mode = "blend"
                group.blend_start = now
            if group.mode == "blend":
                t = 1.0 if self._blend_s <= 0.0 else (now - group.blend_start) / self._blend_s
                if t >= 1.0:
                    group.mode = "track"
                    group.held = {}
                else:
                    s = smoothstep(t)
                    for key, held in group.held.items():
                        if key in action:
                            out[key] = held + s * (action[key] - held)
        self._last_output = out
        return out

    def _ease_offset(self, group: _GroupState, tick_s: float) -> dict[str, float]:
        """Advance a held group's offset toward the desired one and return its current value."""
        target = {key: self._offsets.get(key, 0.0) for key in group.held}
        if target != group.offset_to:
            # The tick that observes a new offset still outputs the current pose (no jump).
            group.offset_from = _offset_at(group)
            group.offset_to = target
            group.offset_progress = 0.0 if self._blend_s > 0.0 else 1.0
        elif group.offset_progress < 1.0:
            group.offset_progress = min(1.0, group.offset_progress + tick_s / self._blend_s)
        return _offset_at(group)


def _offset_at(group: _GroupState) -> dict[str, float]:
    """Offset of a held group at its current ease progress."""
    if group.offset_progress >= 1.0:
        return dict(group.offset_to)
    s = smoothstep(group.offset_progress)
    return {key: start + s * (group.offset_to[key] - start) for key, start in group.offset_from.items()}


class LockFileWatcher:
    """Change detector for the controller-app's lock state file.

    The file is a JSON object ``{"left": bool, "right": bool, "head": bool}`` that the app
    replaces atomically (tmp + rename). ``poll()`` returns the new lock state only when the
    file changed since the last call, ``None`` otherwise. A missing file reports all groups
    unlocked once; a malformed file is logged once and leaves the previous state untouched.

    The object may also carry ``"head_jitter": {"pan": deg, "tilt": deg}``, the offset to
    hold a frozen head at. It is exposed as ``head_jitter`` and refreshed together with the
    lock state; a file without it (or a missing file) means no offset.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._signature: tuple[int, int, int] | None = None
        self._head_jitter = dict.fromkeys(HEAD_JITTER_AXES, 0.0)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def head_jitter(self) -> dict[str, float]:
        """Head offset in degrees per axis (``pan`` / ``tilt``) from the last accepted read."""
        return dict(self._head_jitter)

    def poll(self) -> dict[str, bool] | None:
        try:
            st = os.stat(self._path)
            # st_ino changes on every tmp+rename, so even same-size same-mtime rewrites register.
            signature = (st.st_ino, st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            signature = _MISSING_SIGNATURE
        if signature == self._signature:
            return None
        self._signature = signature
        if signature == _MISSING_SIGNATURE:
            self._head_jitter = dict.fromkeys(HEAD_JITTER_AXES, 0.0)
            return dict.fromkeys(LOCK_GROUPS, False)
        return self._read()

    def _read(self) -> dict[str, bool] | None:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable lock file %s: %s", self._path, e)
            return None
        if not isinstance(data, dict):
            logger.warning("Ignoring lock file %s: expected a JSON object, got %s", self._path, type(data))
            return None
        locks: dict[str, bool] = {}
        for group in LOCK_GROUPS:
            value = data.get(group, False)
            if not isinstance(value, bool):
                logger.warning("Ignoring lock file %s: %r is not a bool for %r", self._path, value, group)
                return None
            locks[group] = value
        head_jitter = self._parse_head_jitter(data.get("head_jitter"))
        if head_jitter is None:
            return None
        self._head_jitter = head_jitter
        return locks

    def _parse_head_jitter(self, value: object) -> dict[str, float] | None:
        """Per-axis degrees clamped to ``MAX_HEAD_JITTER_DEG``; ``None`` (logged) when malformed."""
        if value is None:
            return dict.fromkeys(HEAD_JITTER_AXES, 0.0)
        if not isinstance(value, dict):
            logger.warning("Ignoring lock file %s: head_jitter is not a JSON object", self._path)
            return None
        head_jitter: dict[str, float] = {}
        for axis in HEAD_JITTER_AXES:
            degrees = value.get(axis, 0.0)
            if (
                isinstance(degrees, bool)
                or not isinstance(degrees, (int, float))
                or not math.isfinite(degrees)
            ):
                logger.warning(
                    "Ignoring lock file %s: head_jitter %r is not a finite number for %r",
                    self._path,
                    degrees,
                    axis,
                )
                return None
            head_jitter[axis] = max(-MAX_HEAD_JITTER_DEG, min(MAX_HEAD_JITTER_DEG, float(degrees)))
        return head_jitter
