"""The ``motion.json`` file of a fixed recording, and where fixed recordings live.

``bmh-record-motion`` writes one file per take; the controller-app keeps it next to its own
sidecar (name, trim range) and never edits it:

.. code-block:: text

    ~/.cache/bmh-101/fixed-recordings/<id>/
      motion.json      # this format, written once by the recorder
      bmh.meta.json    # the controller-app's sidecar

Format, version 1 (unknown top-level keys are ignored):

.. code-block:: json

    {"version": 1, "fps": 60,
     "keys": ["left_shoulder_pan.pos", "...", "right_gripper.pos"],
     "frames": [[0.0, 12.3, "..."], ["..."]],
     "recorded_at": 1762600000.0}

``keys`` are the robot's action keys (for the BMH-101: :data:`MOTION_KEYS`), in the order
of every frame's values. One frame per control tick at ``fps``; each frame is the action
actually sent to the follower on that tick. Units as everywhere in the app: degrees for
joints, 0..100 for grippers.

stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

MOTION_VERSION = 1
# Folder of all fixed recordings under the app's cache dir, and the file name inside one.
MOTION_DIR_NAME = "fixed-recordings"
MOTION_FILE_NAME = "motion.json"

# The 16 bimanual action keys of the BMH-101 (`bi_so_follower` with
# `left_arm_config.with_head=True`), in `action_features` order: the left arm's 7 joints,
# the head (on the left bus), then the right arm's 7. The controller-app's `HOME_POSE_KEYS`.
_ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
MOTION_KEYS: tuple[str, ...] = (
    *(f"left_{joint}.pos" for joint in _ARM_JOINTS),
    "left_head_pan.pos",
    "left_head_tilt.pos",
    *(f"right_{joint}.pos" for joint in _ARM_JOINTS),
)


@dataclass(frozen=True)
class MotionFile:
    """One fixed recording.

    Attributes:
        fps: Frames per second the frames were recorded at.
        keys: Action key of each value in a frame.
        frames: One tuple of ``len(keys)`` values per recorded tick (at least one).
        recorded_at: Unix time the recording started.
    """

    fps: int
    keys: tuple[str, ...]
    frames: tuple[tuple[float, ...], ...]
    recorded_at: float

    @property
    def duration_s(self) -> float:
        """Seconds from the first to the last frame."""
        return (len(self.frames) - 1) / self.fps


def _is_number(value: object) -> bool:
    """A finite int / float, not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_motion(data: object) -> MotionFile:
    """Validate a decoded ``motion.json`` document.

    Raises:
        ValueError: Naming the first problem: not an object, ``version`` other than 1,
            ``fps`` not a positive int, ``keys`` not a non-empty list of distinct non-empty
            strings, ``frames`` empty or a frame that is not ``len(keys)`` finite numbers
            (bools rejected), or ``recorded_at`` not a finite number.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    version = data.get("version")
    if type(version) is not int or version != MOTION_VERSION:
        raise ValueError(f"unsupported 'version' {version!r} (expected {MOTION_VERSION})")
    fps = data.get("fps")
    if type(fps) is not int or fps <= 0:
        raise ValueError(f"'fps' must be a positive integer, got {fps!r}")
    keys = data.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("'keys' must be a non-empty list of strings")
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValueError(f"'keys' entries must be non-empty strings, got {key!r}")
    if len(set(keys)) != len(keys):
        raise ValueError("'keys' must not repeat a key")
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("'frames' must be a non-empty list")
    parsed: list[tuple[float, ...]] = []
    for i, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != len(keys):
            raise ValueError(f"frame {i} must be a list of {len(keys)} numbers")
        for value in frame:
            if not _is_number(value):
                raise ValueError(f"frame {i} holds {value!r}, expected a finite number")
        parsed.append(tuple(float(value) for value in frame))
    recorded_at = data.get("recorded_at")
    if (
        isinstance(recorded_at, bool)
        or not isinstance(recorded_at, (int, float))
        or not math.isfinite(recorded_at)
    ):
        raise ValueError(f"'recorded_at' must be a finite number, got {recorded_at!r}")
    return MotionFile(fps=fps, keys=tuple(keys), frames=tuple(parsed), recorded_at=float(recorded_at))


def load_motion(path: Path | str) -> MotionFile:
    """Read and validate a ``motion.json``.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If it is not valid JSON or breaks a rule of :func:`parse_motion`; the
            message names the file.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return parse_motion(json.loads(text))
    except ValueError as e:  # json.JSONDecodeError is a ValueError
        raise ValueError(f"{path}: {e}") from e


def save_motion(path: Path | str, motion: MotionFile) -> None:
    """Write ``motion`` to ``path`` atomically (tmp file + rename), creating the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "version": MOTION_VERSION,
        "fps": motion.fps,
        "keys": list(motion.keys),
        "frames": [list(frame) for frame in motion.frames],
        "recorded_at": motion.recorded_at,
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def motion_root() -> Path:
    """``~/.cache/bmh-101/fixed-recordings`` — the folder of every fixed recording."""
    return Path.home() / ".cache" / "bmh-101" / MOTION_DIR_NAME


def is_under_motion_root(path: Path | str, root: Path | str | None = None) -> bool:
    """Whether ``path`` lies inside ``root`` (default :func:`motion_root`).

    ``..`` and symlinks are resolved first. The root itself is not "under" it.
    """
    base = Path(root if root is not None else motion_root()).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    return resolved != base and resolved.is_relative_to(base)
