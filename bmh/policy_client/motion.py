"""Live motion shaping for the BMH-101 policy client: speed-gated smoothing.

The control loop plays one policy action per tick. The VLA's actions wobble a little from
step to step, which shows as a trembling arm during larger moves. :class:`MotionShaper`
sits between "take the next action from the chunk" and ``robot.send_action`` and applies
a per-joint low-pass behind the Physical Agent's live Smoothing slider (``smoothing_pct``
in the loop file, see ``bmh/policy_client/control.py``).

The low-pass is first order on the commanded position. Its time constant grows with the
joint's *pace* (how fast the raw command moves): fast transport moves are smoothed hard,
while slow fine adjustments keep their small corrective wiggles, so strong smoothing
cannot make the arm miss a precision phase. That is the inverse of a One-Euro filter on
purpose: there, slow motion counts as tremor and fast motion passes to cut lag; here the
slow wiggles are the policy's corrections and the fast ones are noise. ``PACE_FLOOR``
keeps a share of the smoothing at rest.

The filter keeps no velocity. A filter that does (an acceleration limit — "Weight") was
tried and dropped after the robot test: its momentum carries the arm past the policy's
mid-move corrections, and the inference success rate fell sharply even at low settings.
A first-order low-pass lags, but it never overshoots.

Time constants are in seconds and the caller passes the tick's nominal period as ``dt``,
so a playback-speed change does not change how smooth the arm feels.

With the slider at 0 the shaper returns a copy of each target without any arithmetic on
the positions — the default path is bit-identical to the unshaped stream. It still keeps
its bookkeeping then, so a slider moved mid-motion starts from the real pace.

Units: joints are in degrees (``so_follower`` with ``use_degrees=True``), the gripper in
0..100 — one scale for all, good enough. stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

# Tunables that give the slider its scale. Starting guesses, not measured values.
SMOOTHING_TAU_MAX_S = 0.5  # low-pass time constant at smoothing 100 % and full pace
PACE_REF_DEG_S = 60.0  # command pace at (and above) which smoothing is at full strength
PACE_FLOOR = 0.25  # share of the smoothing kept at zero pace (resting jitter is still damped a bit)
PACE_TAU_S = 0.25  # EMA time constant of the pace estimate


def pace_gate(pace: float) -> float:
    """Share of the smoothing applied at ``pace`` (units/s of the raw command).

    ``PACE_FLOOR`` at rest, rising linearly to 1 at ``PACE_REF_DEG_S`` and staying there.
    """
    return PACE_FLOOR + (1.0 - PACE_FLOOR) * min(pace / PACE_REF_DEG_S, 1.0)


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
    command. See the module docstring for what the filter does.
    """

    def __init__(self, smoothing_pct: int):
        """Start without any joint: each one is seeded the first time a command names it.

        Args:
            smoothing_pct: Slider value, 0..100 (0 = off).
        """
        self._smoothing_pct = smoothing_pct
        self._joints: dict[str, _Joint] = {}

    @property
    def smoothing_pct(self) -> int:
        """The slider value in effect."""
        return self._smoothing_pct

    def set_smoothing(self, smoothing_pct: int) -> None:
        """Apply a live slider change from the next :meth:`step` on; the state carries on."""
        self._smoothing_pct = smoothing_pct

    def reset(self, pose: Mapping[str, float] | None = None) -> None:
        """Forget every joint (``None``), or start over with the arms at ``pose``."""
        self._joints = {key: _Joint(float(value)) for key, value in (pose or {}).items()}

    def step(self, target: Mapping[str, float], dt: float) -> dict[str, float]:
        """The command to send this tick for the policy's ``target``.

        Covers exactly ``target``'s keys. A key seen for the first time is passed through
        and seeded there. With the slider at 0 this returns ``dict(target)``.

        Args:
            target: Joint commands of this tick (no ``feedback_*`` keys).
            dt: Nominal tick period in seconds.
        """
        if not dt > 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        off = self._smoothing_pct == 0
        pace_alpha = 1.0 - math.exp(-dt / PACE_TAU_S)
        tau_max = SMOOTHING_TAU_MAX_S * self._smoothing_pct / 100.0
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
            tau = tau_max * pace_gate(joint.pace)
            joint.position += (1.0 - math.exp(-dt / tau)) * (x - joint.position)
            shaped[key] = joint.position
        if off:
            # No arithmetic on the positions: `y + 1.0 * (x - y)` may differ from `x` in
            # the last bit, and the default path must equal the unshaped stream exactly.
            return dict(target)
        return shaped
