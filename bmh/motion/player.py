"""Replay plan for a fixed recording: (present pose, recording, start mode) → ``pose_at(t)``.

Two processes play fixed recordings: ``bmh-replay-motion`` (the controller-app's Fixed
recordings tab, nothing else running) and ``bmh-policy-client`` (a robot script's
``playRecording()`` while the Physical Agent holds the serial ports). Both build one
:class:`ReplayPlan` and send ``plan.pose_at(now - t0)`` every control tick, so the motion is
the same whichever process plays it.

**Time-indexed.** :func:`frame_at` interpolates linearly between the two frames around
``t * fps``. A 60 Hz recording therefore plays at its recorded speed from the policy
client's 30 Hz loop and from the 60 Hz standalone player alike.

**Start modes.** The arms are rarely where the recording starts:

- ``goto-start``: a smoothstep ramp from the present pose to the first kept frame over
  ``ramp_s`` (like the Home ramp), then the recording as is.
- ``blend-in``: the recording starts at once, shifted by the offset between the present
  pose and its first kept frame; that offset fades out with a smoothstep over ``blend_s``.
  The arms start exactly where they are and slide onto the trajectory instead of jumping —
  the same idea as the teleop group lock's offset ease.

After the last frame the plan stays on the final pose (``phase_at`` says ``"done"``); the
players keep sending it until they are told to let go.

stdlib only (no hardware imports), so it is unit-testable anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from bmh.motion.file import MotionFile

Mode = Literal["goto-start", "blend-in"]
MODES: tuple[Mode, ...] = ("goto-start", "blend-in")
Phase = Literal["ramping", "playing", "done"]

# Lead-in of `goto-start` (same as the policy client's `home_ramp_s`) and the offset fade
# of `blend-in`, in seconds.
DEFAULT_RAMP_S = 2.5
DEFAULT_BLEND_S = 1.0


def smoothstep(t: float) -> float:
    """Cubic ease-in/ease-out on [0, 1]; clamps outside that range."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def frame_at(frames: Sequence[Sequence[float]], fps: float, t_s: float) -> tuple[float, ...]:
    """The recording at ``t_s`` seconds: linear between the neighbouring frames.

    Clamped at both ends — before 0 it is the first frame, after the last frame the last.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("no frames")
    position = t_s * fps
    if position <= 0:
        return tuple(frames[0])
    last = len(frames) - 1
    if position >= last:
        return tuple(frames[last])
    i = int(position)
    frac = position - i
    if frac == 0.0:
        return tuple(frames[i])
    return tuple(a + (b - a) * frac for a, b in zip(frames[i], frames[i + 1], strict=True))


@dataclass(frozen=True)
class ReplayPlan:
    """A planned replay of ``frames``, starting at ``t = 0`` from ``start_pose``.

    Build one with :func:`plan_replay`.

    Attributes:
        mode: How the arms get onto the recording (see the module docstring).
        fps: Frame rate of ``frames``.
        keys: Action key of each value, the recording's keys the robot has.
        frames: The kept range of the recording, reduced to ``keys``.
        start_pose: Where the arms were when the replay started, on ``keys``.
        ramp_s: Lead-in of ``goto-start`` (0 for ``blend-in``).
        blend_s: Offset fade of ``blend-in`` (0 for ``goto-start``).
        dropped_keys: The recording's keys the present pose does not have (e.g. a robot
            without a head), left out of every pose.
    """

    mode: Mode
    fps: int
    keys: tuple[str, ...]
    frames: tuple[tuple[float, ...], ...]
    start_pose: tuple[float, ...]
    ramp_s: float
    blend_s: float
    dropped_keys: tuple[str, ...] = ()

    @property
    def play_s(self) -> float:
        """Seconds from the first to the last kept frame."""
        return (len(self.frames) - 1) / self.fps

    @property
    def duration_s(self) -> float:
        """Seconds from the start until the final pose is reached (``phase_at`` → done)."""
        if self.mode == "goto-start":
            return self.ramp_s + self.play_s
        return max(self.play_s, self.blend_s)

    def phase_at(self, t_s: float) -> Phase:
        """``"ramping"`` during the ``goto-start`` lead-in, ``"playing"``, then ``"done"``."""
        if t_s >= self.duration_s:
            return "done"
        if self.mode == "goto-start" and t_s < self.ramp_s:
            return "ramping"
        return "playing"

    def pose_at(self, t_s: float) -> dict[str, float]:
        """The pose to command ``t_s`` seconds after the start, as ``{action key: value}``."""
        first = self.frames[0]
        if self.mode == "goto-start":
            if t_s < self.ramp_s:
                alpha = smoothstep(t_s / self.ramp_s)
                values = tuple(s + (f - s) * alpha for s, f in zip(self.start_pose, first, strict=True))
            else:
                values = frame_at(self.frames, self.fps, t_s - self.ramp_s)
        elif self.blend_s > 0 and t_s <= 0.0:
            values = self.start_pose  # exactly where the arms are, not first + (present - first)
        else:
            values = frame_at(self.frames, self.fps, t_s)
            weight = 1.0 - smoothstep(t_s / self.blend_s) if self.blend_s > 0 else 0.0
            if weight > 0.0:
                values = tuple(
                    v + (s - f) * weight for v, s, f in zip(values, self.start_pose, first, strict=True)
                )
        return dict(zip(self.keys, values, strict=True))


def plan_replay(
    present: Mapping[str, float],
    motion: MotionFile,
    start: int,
    end: int,
    mode: Mode,
    ramp_s: float = DEFAULT_RAMP_S,
    blend_s: float = DEFAULT_BLEND_S,
) -> ReplayPlan:
    """Plan the replay of ``motion.frames[start:end]`` from the ``present`` pose.

    Args:
        present: Where the arms are now (or were last commanded to), as action keys.
        motion: The recording.
        start: First kept frame (the app's trim range, inclusive).
        end: End of the kept frames (exclusive).
        mode: ``"goto-start"`` or ``"blend-in"``.
        ramp_s: Lead-in of ``goto-start``.
        blend_s: Offset fade of ``blend-in``.

    Raises:
        ValueError: On an unknown mode, a range outside the recording, a negative duration,
            or when the present pose has none of the recording's keys.
    """
    if mode not in MODES:
        raise ValueError(f"unknown replay mode {mode!r} (expected one of {', '.join(MODES)})")
    if not 0 <= start < end <= len(motion.frames):
        raise ValueError(f"range [{start}, {end}) is outside the recording's {len(motion.frames)} frames")
    if ramp_s < 0 or blend_s < 0:
        raise ValueError(f"ramp_s and blend_s must be >= 0, got {ramp_s} / {blend_s}")
    keys = tuple(key for key in motion.keys if key in present)
    if not keys:
        raise ValueError("the recording names no joint of this robot")
    dropped = tuple(key for key in motion.keys if key not in present)
    columns = [motion.keys.index(key) for key in keys]
    frames = tuple(tuple(frame[c] for c in columns) for frame in motion.frames[start:end])
    return ReplayPlan(
        mode=mode,
        fps=motion.fps,
        keys=keys,
        frames=frames,
        start_pose=tuple(float(present[key]) for key in keys),
        ramp_s=ramp_s if mode == "goto-start" else 0.0,
        blend_s=blend_s if mode == "blend-in" else 0.0,
        dropped_keys=dropped,
    )
