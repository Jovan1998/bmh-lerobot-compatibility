"""Policy feedback flags for the BMH-101 GR00T client.

A skill trained with *feedbacks* (preset: ``done``) returns one 0..1 value per feedback
with every action step (see ``bmh/groot_client/layout.py::unpack_feedback``).
:class:`FeedbackTracker` turns that noisy per-tick stream into booleans for the
controller-app:

* **hysteresis** — a flag switches on at ``on_threshold`` and off again only at the
  lower ``off_threshold``, so a value hovering around one threshold cannot flap it;
* **debounce** — a switch needs ``debounce_ticks`` *consecutive* ticks beyond the
  threshold; a single tick back inside the band restarts the count.

Flags are level state: :meth:`FeedbackTracker.update` answers with the **full snapshot**
(every key, not a delta) whenever it changed and ``None`` otherwise, so the caller logs
one ``BMH_FEEDBACK`` line per change (see
``bmh/groot_client/control.py::format_feedback_status``) and nothing per tick. The
caller decides *which* ticks count — the control loop only feeds steps of a chunk that
belongs to the current target and calls :meth:`FeedbackTracker.reset` when that changes.

stdlib only (no numpy / zmq / hardware imports) so it is unit-testable anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping


class FeedbackTracker:
    """Hysteresis + consecutive-tick debounce over the policy's feedback values."""

    def __init__(self, on_threshold: float, off_threshold: float, debounce_ticks: int):
        """
        Args:
            on_threshold: A flag that is off counts a tick towards switching on when its
                value is ``>=`` this.
            off_threshold: A flag that is on counts a tick towards switching off when its
                value is ``<=`` this.
            debounce_ticks: Consecutive counted ticks needed to switch, ``>= 1``.

        Raises:
            ValueError: If ``off_threshold > on_threshold`` (the flag would flap) or
                ``debounce_ticks < 1``.
        """
        if off_threshold > on_threshold:
            raise ValueError(f"off_threshold ({off_threshold}) must not exceed on_threshold ({on_threshold})")
        if debounce_ticks < 1:
            raise ValueError(f"debounce_ticks must be >= 1, got {debounce_ticks}")
        self._on = on_threshold
        self._off = off_threshold
        self._debounce = debounce_ticks
        self._flags: dict[str, bool] = {}
        # Consecutive ticks each key's value has spent beyond the threshold that would
        # flip its flag.
        self._streak: dict[str, int] = {}

    def update(self, values: Mapping[str, float]) -> dict[str, bool] | None:
        """Feed one tick of feedback values.

        Args:
            values: ``{feedback key: value}`` of the action step sent this tick — the
                complete key set of the running skill (empty for one without feedbacks).

        Returns:
            The full ``{key: flag}`` snapshot, in ``values`` order, if a flag flipped or
            the key set changed; else ``None``. A key seen for the first time starts off
            and its tick already counts, so with ``debounce_ticks >= 2`` the first
            snapshot is all-false — that is how the app learns which flags exist.
        """
        changed = values.keys() != self._flags.keys()
        if changed:
            self._flags = {key: self._flags.get(key, False) for key in values}
            self._streak = {key: self._streak.get(key, 0) for key in values}
        for key, value in values.items():
            on = self._flags[key]
            # NaN compares false either way: it never counts, and restarts the streak.
            beyond = value <= self._off if on else value >= self._on
            if not beyond:
                self._streak[key] = 0
                continue
            self._streak[key] += 1
            if self._streak[key] >= self._debounce:
                self._flags[key] = not on
                self._streak[key] = 0
                changed = True
        return dict(self._flags) if changed else None

    def reset(self) -> bool:
        """Forget every key, flag and streak (the chunk's owner changed, or motion stopped).

        Returns:
            ``True`` if a non-empty snapshot had been reported — the app then still shows
            flags, and the caller should tell it with an empty snapshot.
        """
        had_flags = bool(self._flags)
        self._flags = {}
        self._streak = {}
        return had_flags
