"""Live motion shaping for the BMH-101 policy client: speed-gated smoothing and weight.

The control loop plays one policy action per tick. Two things about that stream show on
the robot: the VLA's actions wobble a little from step to step (a trembling arm during
larger moves), and consecutive chunks can contradict each other (right, then left, then
a hard correction — the arm whips). :class:`MotionShaper` sits between "take the next
action from the chunk" and ``robot.send_action`` and applies two per-joint filters, each
behind a live slider of the Physical Agent (``smoothing_pct`` / ``weight_pct`` in the
loop file, see ``bmh/policy_client/control.py``):

* **Smoothing** — a first-order low-pass on the commanded position whose time constant
  grows with the joint's *pace* (how fast the raw command moves). Fast transport moves
  are smoothed hard; slow fine adjustments keep their small corrective wiggles, so strong
  smoothing cannot make the arm miss a precision phase. That is the inverse of a
  One-Euro filter on purpose: there, slow motion counts as tremor and fast motion passes
  to cut lag; here the slow wiggles are the policy's corrections and the fast ones are
  noise. ``PACE_FLOOR`` keeps a share of the smoothing at rest.
* **Weight** — an acceleration limit per joint: the commanded velocity changes by at
  most ``a * dt`` per tick, and the joint brakes in time to stop on its target. That is
  a trapezoidal velocity profile, so a reversal decelerates first and a target that
  stays put is never overshot. Slow adjustments pass untouched (their velocity changes
  are far below ``a * dt``). The slider is the time to reach ``FULL_SPEED_DEG_S``:
  about 1 s at 50 %, 2 s at 100 %. That speed is only the slider's scale, never a cap.

Per tick: smoothing first, then weight, so the limiter is not driven by jitter and what
reaches the servos respects the acceleration cap. Time constants are in seconds and the
caller passes the tick's nominal period as ``dt``, so a playback-speed change does not
change how heavy or smooth the arm feels.

With both sliders at 0 the shaper returns a copy of each target without any arithmetic
on the positions — the default path is bit-identical to the unshaped stream. It still
keeps its bookkeeping then, so a slider moved mid-motion starts from the real pace and
velocity.

Units: joints are in degrees (``so_follower`` with ``use_degrees=True``), the gripper in
0..100 — one scale for all, good enough. stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

# Tunables that give the two sliders their scale. All of them are starting guesses, not
# measured values: tune them on the robot.
SMOOTHING_TAU_MAX_S = 0.5  # low-pass time constant at smoothing 100 % and full pace
PACE_REF_DEG_S = 60.0  # command pace at (and above) which smoothing is at full strength
PACE_FLOOR = 0.25  # share of the smoothing kept at zero pace (resting jitter is still damped a bit)
PACE_TAU_S = 0.25  # EMA time constant of the pace estimate
WEIGHT_RAMP_MAX_S = 2.0  # seconds to reach FULL_SPEED_DEG_S at weight 100 % (1 s at 50 %)
# "Full speed" the ramp time refers to — only the slider's scale, the limiter never caps
# velocity. A rough guess at a brisk teleop move; the STS3215 does ~300–375 deg/s unloaded.
FULL_SPEED_DEG_S = 180.0


@dataclass(frozen=True)
class MotionParams:
    """The two motion sliders.

    Attributes:
        smoothing_pct: Speed-gated low-pass strength, 0..100 (0 = off).
        weight_pct: Acceleration limit, 0..100 (0 = off; higher = heavier).
    """

    smoothing_pct: int = 0
    weight_pct: int = 0

    @property
    def off(self) -> bool:
        """Both sliders at 0: the shaper passes every target through untouched."""
        return self.smoothing_pct == 0 and self.weight_pct == 0

    def tau_max_s(self) -> float:
        """Low-pass time constant at full pace, in seconds (0 = no smoothing)."""
        return SMOOTHING_TAU_MAX_S * self.smoothing_pct / 100.0

    def accel_limit(self) -> float | None:
        """Acceleration cap in units/s², ``None`` when weight is off."""
        if self.weight_pct <= 0:
            return None
        return FULL_SPEED_DEG_S / (WEIGHT_RAMP_MAX_S * self.weight_pct / 100.0)


def pace_gate(pace: float) -> float:
    """Share of the smoothing applied at ``pace`` (units/s of the raw command).

    ``PACE_FLOOR`` at rest, rising linearly to 1 at ``PACE_REF_DEG_S`` and staying there.
    """
    return PACE_FLOOR + (1.0 - PACE_FLOOR) * min(pace / PACE_REF_DEG_S, 1.0)


def braking_speed(distance: float, accel: float, dt: float) -> float:
    """Highest speed from which a joint still stops exactly ``distance`` ahead.

    The tick-by-tick version of ``sqrt(2 * accel * distance)``. With ``step = accel * dt``,
    a joint at speed ``v = n * step`` first moves ``v * dt`` this tick and then slows by
    ``step`` per tick, so it needs at least ``step * dt * S(n)`` to stop, where ``S`` is
    the piecewise-linear interpolation of ``n * (n + 1) / 2`` between whole ``n``. This
    returns the ``v`` with ``S(n) = distance / (step * dt)``. A joint that follows it
    slows by exactly ``step`` per tick and its last step lands on the target: no
    overshoot and no oscillation around the target. (The continuous formula would need
    a little more than ``step`` of braking per tick and miss the target by a fraction
    of a step.)

    Args:
        distance: Distance to the target, >= 0.
        accel: Acceleration cap, > 0.
        dt: Tick period in seconds, > 0.
    """
    step = accel * dt
    if distance <= 0.0 or step <= 0.0:
        return 0.0
    budget = distance / (step * dt)  # S(n)
    k = math.floor((math.sqrt(1.0 + 8.0 * budget) - 1.0) / 2.0)
    # Float guard: k must satisfy k(k+1)/2 <= budget < (k+1)(k+2)/2.
    while k > 0 and k * (k + 1) / 2.0 > budget:
        k -= 1
    while (k + 1) * (k + 2) / 2.0 <= budget:
        k += 1
    # On n in [k, k+1]: S(n) = (k+1) n - k(k+1)/2.
    return (budget / (k + 1) + k / 2.0) * step


def limit_acceleration(
    position: float, velocity: float, target: float, accel: float, dt: float
) -> tuple[float, float]:
    """One tick of the weight limiter: move ``position`` towards ``target``.

    The new velocity stays within ``velocity ± accel * dt``. Towards the target it is also
    capped by :func:`braking_speed`, unless braking that hard would itself exceed the
    acceleration cap (the target jumped closer than the joint can stop): then the joint
    brakes at the cap and overshoots, like a mass would. When the target can be reached
    this tick within both caps, the joint lands on it exactly (no float residue).

    Returns:
        ``(new position, new velocity)``; the velocity is always the step just taken
        divided by ``dt``.
    """
    distance = target - position
    wanted = distance / dt
    step = accel * dt
    low, high = velocity - step, velocity + step
    if distance > 0.0:
        high = max(min(high, braking_speed(distance, accel, dt)), low)
    elif distance < 0.0:
        low = min(max(low, -braking_speed(-distance, accel, dt)), high)
    # Landing tolerance: on the last braking tick `wanted` and the band edge are equal up
    # to float rounding; stepping by the edge instead would overshoot by an ulp.
    slack = 1e-9 * step
    if low - slack <= wanted <= high + slack:
        return target, wanted
    new_velocity = min(max(wanted, low), high)
    return position + new_velocity * dt, new_velocity


class _Joint:
    """Per-joint shaper state."""

    __slots__ = ("pace", "position", "raw", "velocity")

    def __init__(self, position: float):
        self.position = position  # last output position (what the servo was told)
        self.velocity = 0.0  # last output velocity, units/s
        self.pace = 0.0  # EMA of the raw command's speed, units/s
        self.raw = position  # last raw command (pace input)


class MotionShaper:
    """Speed-gated smoothing and an acceleration limit on the commanded joint positions.

    Keeps per-joint state: the last output position and velocity, the pace EMA and the
    last raw command. See the module docstring for what the two filters do.
    """

    def __init__(self, params: MotionParams):
        """Start without any joint: each one is seeded the first time a command names it."""
        self._params = params
        self._joints: dict[str, _Joint] = {}

    @property
    def params(self) -> MotionParams:
        """The sliders in effect."""
        return self._params

    def set_params(self, params: MotionParams) -> None:
        """Apply a live slider change from the next :meth:`step` on; the state carries on."""
        self._params = params

    def reset(self, pose: Mapping[str, float] | None = None) -> None:
        """Forget every joint (``None``), or start over with the robot at rest at ``pose``."""
        self._joints = {key: _Joint(float(value)) for key, value in (pose or {}).items()}

    def hold(self) -> None:
        """A tick on which the commanded pose stays put: every joint's velocity is 0."""
        for joint in self._joints.values():
            joint.velocity = 0.0

    def step(self, target: Mapping[str, float], dt: float) -> dict[str, float]:
        """The command to send this tick for the policy's ``target``.

        Covers exactly ``target``'s keys. A key seen for the first time is passed through
        and seeded at rest there. With both sliders at 0 this returns ``dict(target)``.

        Args:
            target: Joint commands of this tick (no ``feedback_*`` keys).
            dt: Nominal tick period in seconds.
        """
        if not dt > 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        params = self._params
        pace_alpha = 1.0 - math.exp(-dt / PACE_TAU_S)
        tau_max = params.tau_max_s()
        accel = params.accel_limit()
        shaped: dict[str, float] = {}
        for key, value in target.items():
            x = float(value)
            joint = self._joints.get(key)
            if joint is None:
                self._joints[key] = _Joint(x)
                shaped[key] = value
                continue
            # 1. Pace, from the raw command: the output must not gate its own filter.
            joint.pace += (abs(x - joint.raw) / dt - joint.pace) * pace_alpha
            joint.raw = x
            if params.off:
                # Bookkeeping only; the value goes out untouched (see below).
                joint.velocity = (x - joint.position) / dt
                joint.position = x
                continue
            # 2. Smoothing, stronger at pace.
            tau = tau_max * pace_gate(joint.pace)
            smoothed = (
                x if tau == 0.0 else joint.position + (1.0 - math.exp(-dt / tau)) * (x - joint.position)
            )
            # 3. Weight.
            if accel is None:
                joint.velocity = (smoothed - joint.position) / dt
                joint.position = smoothed
            else:
                joint.position, joint.velocity = limit_acceleration(
                    joint.position, joint.velocity, smoothed, accel, dt
                )
            shaped[key] = joint.position
        if params.off:
            # No arithmetic on the positions: `y + 1.0 * (x - y)` may differ from `x` in
            # the last bit, and the default path must equal the unshaped stream exactly.
            return dict(target)
        return shaped
