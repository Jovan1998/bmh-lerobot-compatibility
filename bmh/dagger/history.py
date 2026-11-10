"""Rewind history of a DAgger session: the last actions sent to the follower.

A ring buffer the control loop appends to while the robot is being moved (policy, teleop);
a tick that only holds a pose does not push. The session addresses it with a *cursor*:
frames back from the newest one, 0 = newest. Stepping moves the cursor one frame per tick,
so the robot replays its own motion; running or recording from a rewound cursor first
drops everything newer (`truncate`), so the history stays one continuous motion.

stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping


class ActionHistory:
    """The newest ``max_frames`` actions sent, oldest dropped first."""

    def __init__(self, max_frames: int):
        """Start empty.

        Raises:
            ValueError: If ``max_frames`` is below 1.
        """
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        self._frames: deque[dict[str, float]] = deque(maxlen=max_frames)

    def __len__(self) -> int:
        """Frames held."""
        return len(self._frames)

    def push(self, action: Mapping[str, float]) -> None:
        """Append a copy of ``action`` as the newest frame."""
        self._frames.append(dict(action))

    def at(self, cursor: int) -> dict[str, float]:
        """A copy of the frame ``cursor`` frames back from the newest (0 = newest).

        Raises:
            IndexError: If ``cursor`` is negative or reaches past the oldest frame.
        """
        if not 0 <= cursor < len(self._frames):
            raise IndexError(f"cursor {cursor} is outside the history ({len(self._frames)} frames)")
        return dict(self._frames[-1 - cursor])

    def truncate(self, cursor: int) -> None:
        """Drop everything newer than the frame at ``cursor``, which becomes the newest.

        Raises:
            IndexError: As `at`.
        """
        if not 0 <= cursor < len(self._frames):
            raise IndexError(f"cursor {cursor} is outside the history ({len(self._frames)} frames)")
        for _ in range(cursor):
            self._frames.pop()

    def clear(self) -> None:
        """Forget every frame."""
        self._frames.clear()
