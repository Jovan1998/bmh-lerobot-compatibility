"""Unit tests for ``bmh.policy_client.motion`` — the speed-gated smoothing and the weight
(acceleration limit) the policy client applies between the chunk and the servos.

Pure stdlib module, no optional deps. Synthetic series at 30 Hz: the default path must
return the input bit for bit; smoothing must follow a first-order step response whose
time constant grows with the raw command's pace; weight must keep every per-tick velocity
change within ``a * dt``, never overshoot a target that stays put, land on it exactly and
decelerate before it reverses. Run with ``uv run --no-sync pytest tests/bmh/test_motion_shaper.py -q``.
"""

import math
import random

import pytest

from bmh.policy_client.motion import (
    FULL_SPEED_DEG_S,
    PACE_FLOOR,
    PACE_REF_DEG_S,
    PACE_TAU_S,
    SMOOTHING_TAU_MAX_S,
    MotionParams,
    MotionShaper,
    braking_speed,
    pace_gate,
)

DT = 1 / 30
EPS = 1e-9


def _run(shaper: MotionShaper, targets: list[float], key: str = "j") -> list[float]:
    return [shaper.step({key: x}, DT)[key] for x in targets]


def _velocities(positions: list[float]) -> list[float]:
    return [(b - a) / DT for a, b in zip(positions, positions[1:], strict=False)]


def _max_dv(positions: list[float]) -> float:
    v = _velocities(positions)
    return max(abs(b - a) for a, b in zip(v, v[1:], strict=False))


# --------------------------------------------------------------------------- params


def test_params_scale():
    assert MotionParams().off and MotionParams(0, 0).off
    assert not MotionParams(1, 0).off and not MotionParams(0, 1).off
    assert MotionParams(0, 0).tau_max_s() == 0.0
    assert MotionParams(100, 0).tau_max_s() == pytest.approx(SMOOTHING_TAU_MAX_S)
    assert MotionParams(50, 0).tau_max_s() == pytest.approx(SMOOTHING_TAU_MAX_S / 2)
    assert MotionParams(0, 0).accel_limit() is None
    # The slider is the time to full speed: ~1 s at 50 %, ~2 s at 100 %.
    assert FULL_SPEED_DEG_S / MotionParams(0, 50).accel_limit() == pytest.approx(1.0)
    assert FULL_SPEED_DEG_S / MotionParams(0, 100).accel_limit() == pytest.approx(2.0)


# --------------------------------------------------------------------------- off


def test_off_returns_the_input_bit_for_bit():
    rng = random.Random(7)
    shaper = MotionShaper(MotionParams(0, 0))
    for _ in range(2000):
        target = {"a": rng.uniform(-180, 180), "b": rng.gauss(0, 1e-3) + 1 / 3, "c": rng.random() * 100}
        out = shaper.step(target, DT)
        assert out == target and out is not target
        assert all(out[k] is target[k] for k in target)  # the very same float objects


def test_off_covers_exactly_the_target_keys():
    shaper = MotionShaper(MotionParams(0, 0))
    shaper.reset({"a": 1.0, "b": 2.0, "other": 3.0})
    assert shaper.step({"a": 5.0}, DT) == {"a": 5.0}
    assert shaper.step({"a": 5.0, "new": 7.0}, DT) == {"a": 5.0, "new": 7.0}


def test_on_covers_exactly_the_target_keys():
    shaper = MotionShaper(MotionParams(50, 50))
    shaper.reset({"a": 1.0, "other": 3.0})
    assert set(shaper.step({"a": 5.0, "new": 7.0}, DT)) == {"a", "new"}


def test_non_positive_dt_is_refused():
    with pytest.raises(ValueError, match="dt"):
        MotionShaper(MotionParams(50, 0)).step({"a": 1.0}, 0.0)


# --------------------------------------------------------------------------- smoothing


def test_pace_gate_is_monotonic_from_the_floor_to_one():
    paces = [i * 0.5 for i in range(400)]
    gates = [pace_gate(p) for p in paces]
    assert all(b >= a for a, b in zip(gates, gates[1:], strict=False))
    assert pace_gate(0.0) == PACE_FLOOR
    assert pace_gate(PACE_REF_DEG_S) == 1.0 and pace_gate(10 * PACE_REF_DEG_S) == 1.0


def test_smoothing_step_response_at_full_pace():
    # A 90° step drives the pace estimate far above PACE_REF_DEG_S for the first ~13 ticks,
    # so the low-pass runs at its full time constant: 0.5 s at 100 %.
    shaper = MotionShaper(MotionParams(100, 0))
    shaper.reset({"j": 0.0})
    out = _run(shaper, [90.0] * 10)
    expected = [90.0 * (1 - math.exp(-k * DT / SMOOTHING_TAU_MAX_S)) for k in range(1, 11)]
    assert out == pytest.approx(expected, rel=1e-9)


def test_smoothing_step_response_at_rest_uses_the_floor():
    # A micro step barely moves the pace: the time constant is the floor's, 0.125 s.
    shaper = MotionShaper(MotionParams(100, 0))
    shaper.reset({"j": 0.0})
    step = 1e-3
    out = _run(shaper, [step] * 20)
    tau = SMOOTHING_TAU_MAX_S * PACE_FLOOR
    expected = [step * (1 - math.exp(-k * DT / tau)) for k in range(1, 21)]
    assert out == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("speed", [6.0, 30.0, 120.0])
def test_smoothing_lag_on_a_ramp_follows_the_pace_gate(speed):
    # Steady state of y += alpha * (x - y) on a ramp of `speed`: the output trails by
    # speed * dt / (exp(dt / tau) - 1), with tau set by the ramp's own pace.
    shaper = MotionShaper(MotionParams(100, 0))
    shaper.reset({"j": 0.0})
    targets = [speed * DT * k for k in range(1, 300)]
    out = _run(shaper, targets)
    tau = SMOOTHING_TAU_MAX_S * pace_gate(speed)
    lag = speed * DT / (math.exp(DT / tau) - 1)
    assert targets[-1] - out[-1] == pytest.approx(lag, rel=1e-6)


def test_smoothing_lets_slow_corrections_through_better_than_fast_moves():
    # Lag relative to speed: a slow adjustment trails by much less time than a fast move.
    def lag_s(speed: float) -> float:
        shaper = MotionShaper(MotionParams(100, 0))
        shaper.reset({"j": 0.0})
        targets = [speed * DT * k for k in range(1, 300)]
        return (targets[-1] - _run(shaper, targets)[-1]) / speed

    assert lag_s(3.0) < 0.4 * lag_s(120.0)


def test_pace_is_measured_on_the_raw_command():
    # Same raw series, smoothing on vs off: the pace — and therefore the strength — must
    # not depend on what the filter output did. Switch smoothing on after a fast ramp: the
    # very next step already runs at full strength.
    shaper = MotionShaper(MotionParams(0, 0))
    shaper.reset({"j": 0.0})
    _run(shaper, [120.0 * DT * k for k in range(1, 60)])
    shaper.set_params(MotionParams(100, 0))
    y0 = 120.0 * DT * 59
    x = y0 + 120.0 * DT
    out = shaper.step({"j": x}, DT)["j"]
    alpha = 1 - math.exp(-DT / SMOOTHING_TAU_MAX_S)
    assert out == pytest.approx(y0 + alpha * (x - y0), rel=1e-12)
    assert PACE_TAU_S > 0  # the pace EMA's constant is a module tunable


# --------------------------------------------------------------------------- weight


@pytest.mark.parametrize("weight", [50, 100])
def test_weight_step_is_a_trapezoid_that_lands_exactly(weight):
    params = MotionParams(0, weight)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"j": 0.0})
    out = [0.0, *_run(shaper, [90.0] * 200)]
    assert _max_dv(out) <= a * DT + EPS  # never more than a·dt of velocity change per tick
    assert max(out) <= 90.0 + 1e-6  # never overshoots
    assert out[-1] == 90.0  # lands exactly
    settled = out.index(90.0)
    assert all(y == 90.0 for y in out[settled:])
    # ...at rest: a further step does not move.
    assert shaper.step({"j": 90.0}, DT) == {"j": 90.0}
    # Triangular profile: about 2·sqrt(d / a) to get there.
    assert abs(settled * DT - 2 * math.sqrt(90.0 / a)) <= DT + EPS
    # Monotonic approach, accelerating then braking.
    assert all(b >= a_ for a_, b in zip(out, out[1:], strict=False))


def test_weight_reversal_decelerates_first():
    params = MotionParams(0, 100)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"j": 0.0})
    out = [0.0, *_run(shaper, [90.0] * 20)]
    v_at_switch = _velocities(out)[-1]
    assert v_at_switch == pytest.approx(20 * a * DT)  # still accelerating
    out += _run(shaper, [-90.0] * 300)
    v = _velocities(out)
    assert _max_dv(out) <= a * DT + EPS
    # The joint keeps moving forward while it brakes: the velocity changes sign only after
    # passing through (near) zero, never by a jump.
    first_negative = next(i for i, vi in enumerate(v) if vi < 0)
    assert first_negative > 20
    assert all(vi >= 0 for vi in v[:first_negative])
    assert 0 <= v[first_negative - 1] <= a * DT + EPS
    # The turn point is a braking distance (v² / 2a) beyond the switch point.
    turn = max(out)
    assert turn == pytest.approx(out[20] + v_at_switch**2 / (2 * a), abs=v_at_switch * DT)
    assert out[-1] == -90.0 and min(out) >= -90.0 - 1e-6


def test_weight_leaves_slow_adjustments_untouched():
    # Speeds and speed changes within a·dt per tick are passed through exactly.
    params = MotionParams(0, 100)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"j": 10.0})
    speed = 0.9 * a * DT
    targets = [10.0 + 2.0 * math.sin(speed / 2.0 * DT * k) for k in range(1, 200)]
    assert _run(shaper, targets) == targets


def test_weight_caps_acceleration_on_a_jittery_series_with_jumps():
    rng = random.Random(3)
    for smoothing in (0, 60):
        params = MotionParams(smoothing, 100)
        shaper = MotionShaper(params)
        x, targets = 0.0, []
        for _ in range(5000):
            x += rng.uniform(-3, 3) if rng.random() < 0.95 else rng.uniform(-60, 60)
            targets.append(x)
        out = _run(shaper, targets)
        assert _max_dv(out) <= params.accel_limit() * DT + 1e-6


def test_braking_speed_lands_on_whole_tick_counts():
    accel = 90.0
    step = accel * DT
    for n in range(0, 40):
        distance = step * DT * n * (n + 1) / 2
        assert braking_speed(distance, accel, DT) == pytest.approx(n * step, rel=1e-9, abs=1e-12)
    assert braking_speed(0.0, accel, DT) == 0.0
    assert braking_speed(-1.0, accel, DT) == 0.0
    # Between whole counts it is the piecewise-linear interpolation, monotonic in distance.
    speeds = [braking_speed(d / 10, accel, DT) for d in range(0, 2000)]
    assert all(b >= a for a, b in zip(speeds, speeds[1:], strict=False))
    # ...and below the continuous sqrt(2 a d) (it accounts for the discrete ticks).
    assert all(braking_speed(d, accel, DT) <= math.sqrt(2 * accel * d) + EPS for d in (0.1, 1.0, 10.0, 90.0))


# --------------------------------------------------------------------------- state


def test_hold_zeroes_the_velocity():
    params = MotionParams(0, 100)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"j": 0.0})
    out = _run(shaper, [90.0] * 20)
    shaper.hold()
    nxt = shaper.step({"j": 90.0}, DT)["j"]
    # From rest again: one tick of acceleration, not the 20 ticks' worth it had.
    assert nxt - out[-1] == pytest.approx(a * DT * DT)


def test_reset_with_a_pose_starts_at_rest_there():
    params = MotionParams(0, 100)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"j": 0.0})
    _run(shaper, [90.0] * 20)
    shaper.reset({"j": 10.0})
    assert shaper.step({"j": 50.0}, DT)["j"] == pytest.approx(10.0 + a * DT * DT)


def test_reset_without_a_pose_forgets_everything():
    shaper = MotionShaper(MotionParams(0, 100))
    shaper.reset({"j": 0.0})
    shaper.reset()
    assert shaper.step({"j": 50.0}, DT) == {"j": 50.0}  # first sight: seeded there


def test_a_new_key_is_seeded_without_a_jump():
    params = MotionParams(80, 100)
    a = params.accel_limit()
    shaper = MotionShaper(params)
    shaper.reset({"a": 0.0})
    assert shaper.step({"a": 0.0, "b": 50.0}, DT)["b"] == 50.0
    assert shaper.step({"a": 0.0, "b": 60.0}, DT)["b"] - 50.0 <= a * DT * DT + EPS


def test_set_params_applies_from_the_next_step_and_keeps_the_velocity():
    # A ramp at 30 deg/s with shaping off, then weight 100: the limiter starts from the
    # real 30 deg/s (braking at a·dt), not from rest.
    shaper = MotionShaper(MotionParams(0, 0))
    shaper.reset({"j": 0.0})
    targets = [30.0 * DT * k for k in range(1, 31)]
    assert _run(shaper, targets) == targets
    params = MotionParams(0, 100)
    shaper.set_params(params)
    assert shaper.params == params
    nxt = shaper.step({"j": 30.0 * DT * 31}, DT)["j"]
    assert (nxt - targets[-1]) / DT == pytest.approx(30.0 - params.accel_limit() * DT)
