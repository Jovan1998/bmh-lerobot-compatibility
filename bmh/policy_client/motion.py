"""Live motion shaping for the BMH-101 policy client: look-ahead and speed-gated smoothing.

The control loop plays one policy action per tick. The VLA's actions wobble a little from
step to step, which shows as a trembling arm during larger moves. Two stages sit between
"take the next action from the chunk" and ``robot.send_action``, in this order, each
behind a live slider of the Physical Agent (loop file keys, see
``bmh/policy_client/control.py``):

1. :class:`LookaheadSmoother` (``lookahead_frames``): a centred average over the actions
   just played and the ones the current chunk has already planned. Centred, so it does
   not lag: a steady move comes out on its path. It needs the chunk, so it runs first.
2. :class:`MotionShaper` (``smoothing_pct``): a per-joint low-pass on the stream. It sees
   only the past, so it lags behind a moving command (about 33 ms per 10 %), in exchange
   for smoothing what the look-ahead cannot see: plan changes at chunk seams and ticks
   where the chunk's future has run out.

**Look-ahead.** With ``N`` frames the action played at a tick is a Gaussian-weighted
average (sigma ``N/2``) of the last ``h`` played commands, the command itself and the
next ``h`` planned ones, ``h = min(N, played, planned)``. The window shrinks the same on
both sides when either runs short (the end of a chunk, a late chunk, just after a
clear), so it smooths less there instead of lagging. All weights are positive, so the
output stays inside the range of the commands it averages. In frames, not seconds: the
jitter is per action, so the same ``N`` removes the same jitter at any playback speed.

**Smoothing.** First order on the commanded position. Its time constant grows with the
joint's *pace* (how fast the raw command moves): fast transport moves are smoothed hard,
while slow fine adjustments keep their small corrective wiggles, so strong smoothing
cannot make the arm miss a precision phase. That is the inverse of a One-Euro filter on
purpose: there, slow motion counts as tremor and fast motion passes to cut lag; here the
slow wiggles are the policy's corrections and the fast ones are noise. ``PACE_FLOOR``
keeps a share of the smoothing at rest. Time constants are in seconds and the caller
passes the tick's nominal period as ``dt``, so a playback-speed change does not change
how smooth the arm feels.

A low-pass that trails a moving command would jump to it if the slider dropped at once.
So the time constant may only fall at a bounded rate (``SMOOTHING_EASE_S``): lowering
the slider, or switching it off, eases the arm onto the command over a fraction of a
second. Raising it applies at once.

Neither stage keeps a velocity. A filter that does (an acceleration limit — "Weight") was
tried and dropped after the robot test: its momentum carries the arm past the policy's
mid-move corrections, and the inference success rate fell sharply even at low settings.

**Off.** Each stage at 0 (the smoothing once its ease-out has finished) returns a copy of
each target without any arithmetic on the positions, so with both at 0 the stream is
bit-identical to the unshaped one. Both still keep their bookkeeping then, so a slider
moved mid-motion starts from the real pace and a full look-ahead window.

Units: joints are in degrees (``so_follower`` with ``use_degrees=True``), the gripper in
0..100 — one scale for all, good enough. stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence

# Tunables that give the Smoothing slider its scale. Starting guesses, not measured values.
SMOOTHING_TAU_MAX_S = 0.5  # low-pass time constant at smoothing 100 % and full pace
PACE_REF_DEG_S = 60.0  # command pace at (and above) which smoothing is at full strength
PACE_FLOOR = 0.25  # share of the smoothing kept at zero pace (resting jitter is still damped a bit)
PACE_TAU_S = 0.25  # EMA time constant of the pace estimate
# How fast the time constant may fall after the slider was lowered: it decays with this
# time constant and snaps to the slider's value below `_EASE_SNAP_S`. At 60 deg/s,
# 30 % -> off moves at most 3.2 deg in a tick (instead of 10) and is exact after 0.8 s.
SMOOTHING_EASE_S = 0.15
_EASE_SNAP_S = 1e-3

# Most frames the look-ahead averages on each side (the loop file's upper bound).
LOOKAHEAD_MAX_FRAMES = 10


def pace_gate(pace: float) -> float:
    """Share of the smoothing applied at ``pace`` (units/s of the raw command).

    ``PACE_FLOOR`` at rest, rising linearly to 1 at ``PACE_REF_DEG_S`` and staying there.
    """
    return PACE_FLOOR + (1.0 - PACE_FLOOR) * min(pace / PACE_REF_DEG_S, 1.0)


def _smoothing_tau_max(smoothing_pct: int) -> float:
    return SMOOTHING_TAU_MAX_S * smoothing_pct / 100.0


class _Joint:
    """Per-joint shaper state."""

    __slots__ = ("pace", "position", "raw")

    def __init__(self, position: float):
        self.position = position  # last output position (what the servo was told)
        self.pace = 0.0  # EMA of the raw command's speed, units/s
        self.raw = position  # last raw command (pace input)


class MotionShaper:
    """Speed-gated smoothing of the commanded joint positions.

    Keeps per-joint state: the last output position, the pace EMA and the last raw
    command, plus the time constant in effect (which eases down after the slider was
    lowered). See the module docstring for what the filter does.
    """

    def __init__(self, smoothing_pct: int):
        """Start without any joint: each one is seeded the first time a command names it.

        Args:
            smoothing_pct: Slider value, 0..100 (0 = off).
        """
        self._smoothing_pct = smoothing_pct
        # Time constant at full pace in effect; trails the slider's value only downwards.
        self._tau_max = _smoothing_tau_max(smoothing_pct)
        self._joints: dict[str, _Joint] = {}

    @property
    def smoothing_pct(self) -> int:
        """The slider value in effect."""
        return self._smoothing_pct

    @property
    def easing(self) -> bool:
        """Whether a lowered slider is still being eased in (the filter is stronger than set)."""
        return self._tau_max > _smoothing_tau_max(self._smoothing_pct)

    def set_smoothing(self, smoothing_pct: int) -> None:
        """Apply a live slider change from the next :meth:`step` on; the state carries on.

        A higher value applies at once. A lower one (also 0) eases in, so an arm that
        trails a moving command is not snapped onto it.
        """
        self._smoothing_pct = smoothing_pct

    def reset(self, pose: Mapping[str, float] | None = None) -> None:
        """Forget every joint (``None``), or start over with the arms at ``pose``.

        Nothing trails a command afterwards, so the slider's value applies at once.
        """
        self._joints = {key: _Joint(float(value)) for key, value in (pose or {}).items()}
        self._tau_max = _smoothing_tau_max(self._smoothing_pct)

    def step(self, target: Mapping[str, float], dt: float) -> dict[str, float]:
        """The command to send this tick for the policy's ``target``.

        Covers exactly ``target``'s keys. A key seen for the first time is passed through
        and seeded there. With the slider at 0 (and nothing left to ease in) this returns
        ``dict(target)``.

        Args:
            target: Joint commands of this tick (no ``feedback_*`` keys).
            dt: Nominal tick period in seconds.
        """
        if not dt > 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        goal = _smoothing_tau_max(self._smoothing_pct)
        if goal >= self._tau_max:
            self._tau_max = goal
        else:
            self._tau_max = max(goal, self._tau_max * math.exp(-dt / SMOOTHING_EASE_S))
            if self._tau_max < _EASE_SNAP_S:
                self._tau_max = goal
        off = self._tau_max == 0.0
        pace_alpha = 1.0 - math.exp(-dt / PACE_TAU_S)
        shaped: dict[str, float] = {}
        for key, value in target.items():
            x = float(value)
            joint = self._joints.get(key)
            if joint is None:
                self._joints[key] = _Joint(x)
                shaped[key] = value
                continue
            # Pace, from the raw command: the output must not gate its own filter.
            joint.pace += (abs(x - joint.raw) / dt - joint.pace) * pace_alpha
            joint.raw = x
            if off:
                # Bookkeeping only; the value goes out untouched (see below).
                joint.position = x
                continue
            tau = self._tau_max * pace_gate(joint.pace)
            joint.position += (1.0 - math.exp(-dt / tau)) * (x - joint.position)
            shaped[key] = joint.position
        if off:
            # No arithmetic on the positions: `y + 1.0 * (x - y)` may differ from `x` in
            # the last bit, and the default path must equal the unshaped stream exactly.
            return dict(target)
        return shaped


class LookaheadSmoother:
    """Centred smoothing over the actions just played and the ones the chunk has planned.

    Keeps the last :data:`LOOKAHEAD_MAX_FRAMES` raw commands it was given. See the module
    docstring for what it does.
    """

    def __init__(self, frames: int):
        """Start with no history: the window grows as commands are played.

        Args:
            frames: Slider value, 0..``LOOKAHEAD_MAX_FRAMES`` frames on each side (0 = off).
        """
        self._frames = frames
        self._past: deque[dict[str, float]] = deque(maxlen=LOOKAHEAD_MAX_FRAMES)
        self._weights: dict[tuple[int, int], tuple[float, ...]] = {}

    @property
    def frames(self) -> int:
        """The slider value in effect: how many planned actions :meth:`step` wants to see."""
        return self._frames

    def set_frames(self, frames: int) -> None:
        """Apply a live slider change from the next :meth:`step` on; the history carries on."""
        self._frames = frames

    def clear(self) -> None:
        """Forget the played commands: the next ones do not follow on from them.

        For every tick that does not play a chunk action (Home ramp, hold, late chunk).
        """
        self._past.clear()

    def step(self, target: Mapping[str, float], upcoming: Sequence[Mapping[str, float]]) -> dict[str, float]:
        """The command to play this tick for the chunk action ``target``.

        Covers exactly ``target``'s keys. With 0 frames, or nothing to average on one side,
        this returns ``dict(target)``. ``target`` joins the history in every case.

        Args:
            target: Joint commands of this tick (no ``feedback_*`` keys).
            upcoming: The chunk's next actions, joints only, nearest first. Only the first
                :attr:`frames` are read; an entry missing one of ``target``'s keys ends the
                usable future.
        """
        if self._past and set(self._past[-1]) != set(target):
            self._past.clear()  # other joints: the history does not describe this stream
        h = min(self._frames, len(self._past))
        future = 0
        for entry in upcoming[:h]:
            if not all(key in entry for key in target):
                break
            future += 1
        h = min(h, future)
        window: list[Mapping[str, float]] = []
        if h:
            window = [*list(self._past)[-h:], target, *upcoming[:h]]
        self._past.append(dict(target))
        if not h:
            return dict(target)
        weights = self._weights_for(h)
        return {
            key: sum(w * float(entry[key]) for w, entry in zip(weights, window, strict=True))
            for key in target
        }

    def _weights_for(self, h: int) -> tuple[float, ...]:
        """Normalised Gaussian weights for offsets -h..h, sigma = frames / 2."""
        cache_key = (self._frames, h)
        weights = self._weights.get(cache_key)
        if weights is None:
            sigma = self._frames / 2.0
            raw = [math.exp(-(k * k) / (2.0 * sigma * sigma)) for k in range(-h, h + 1)]
            total = sum(raw)
            weights = tuple(w / total for w in raw)
            self._weights[cache_key] = weights
        return weights
