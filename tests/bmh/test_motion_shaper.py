"""Unit tests for ``bmh.policy_client.motion`` — the speed-gated smoothing the policy client
applies between the chunk and the servos.

Pure stdlib module, no optional deps. Synthetic series at 30 Hz: the default path must
return the input bit for bit; smoothing must follow a first-order step response whose
time constant grows with the raw command's pace, and must never carry the arm past a
target (no momentum). Run with ``uv run --no-sync pytest tests/bmh/test_motion_shaper.py -q``.
"""

import math
import random

import pytest

from bmh.policy_client.motion import (
    PACE_FLOOR,
    PACE_REF_DEG_S,
    PACE_TAU_S,
    SMOOTHING_TAU_MAX_S,
    MotionShaper,
    pace_gate,
)

DT = 1 / 30
EPS = 1e-9


def _run(shaper: MotionShaper, targets: list[float], key: str = "j") -> list[float]:
    return [shaper.step({key: x}, DT)[key] for x in targets]


# --------------------------------------------------------------------------- off


def test_off_returns_the_input_bit_for_bit():
    rng = random.Random(7)
    shaper = MotionShaper(0)
    for _ in range(2000):
        target = {"a": rng.uniform(-180, 180), "b": rng.gauss(0, 1e-3) + 1 / 3, "c": rng.random() * 100}
        out = shaper.step(target, DT)
        assert out == target and out is not target
        assert all(out[k] is target[k] for k in target)  # the very same float objects


def test_off_covers_exactly_the_target_keys():
    shaper = MotionShaper(0)
    shaper.reset({"a": 1.0, "b": 2.0, "other": 3.0})
    assert shaper.step({"a": 5.0}, DT) == {"a": 5.0}
    assert shaper.step({"a": 5.0, "new": 7.0}, DT) == {"a": 5.0, "new": 7.0}


def test_on_covers_exactly_the_target_keys():
    shaper = MotionShaper(50)
    shaper.reset({"a": 1.0, "other": 3.0})
    assert set(shaper.step({"a": 5.0, "new": 7.0}, DT)) == {"a", "new"}


def test_non_positive_dt_is_refused():
    with pytest.raises(ValueError, match="dt"):
        MotionShaper(50).step({"a": 1.0}, 0.0)


# --------------------------------------------------------------------------- smoothing


def test_pace_gate_is_monotonic_from_the_floor_to_one():
    paces = [i * 0.5 for i in range(400)]
    gates = [pace_gate(p) for p in paces]
    assert all(b >= a for a, b in zip(gates, gates[1:], strict=False))
    assert pace_gate(0.0) == PACE_FLOOR
    assert pace_gate(PACE_REF_DEG_S) == 1.0 and pace_gate(10 * PACE_REF_DEG_S) == 1.0


@pytest.mark.parametrize("smoothing", [50, 100])
def test_smoothing_step_response_at_full_pace(smoothing):
    # A 90° step drives the pace estimate far above PACE_REF_DEG_S for the first ~13 ticks,
    # so the low-pass runs at its full time constant: 0.5 s at 100 %, 0.25 s at 50 %.
    shaper = MotionShaper(smoothing)
    shaper.reset({"j": 0.0})
    out = _run(shaper, [90.0] * 10)
    tau = SMOOTHING_TAU_MAX_S * smoothing / 100
    expected = [90.0 * (1 - math.exp(-k * DT / tau)) for k in range(1, 11)]
    assert out == pytest.approx(expected, rel=1e-9)


def test_smoothing_step_response_at_rest_uses_the_floor():
    # A micro step barely moves the pace: the time constant is the floor's, 0.125 s.
    shaper = MotionShaper(100)
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
    shaper = MotionShaper(100)
    shaper.reset({"j": 0.0})
    targets = [speed * DT * k for k in range(1, 300)]
    out = _run(shaper, targets)
    tau = SMOOTHING_TAU_MAX_S * pace_gate(speed)
    lag = speed * DT / (math.exp(DT / tau) - 1)
    assert targets[-1] - out[-1] == pytest.approx(lag, rel=1e-6)


def test_smoothing_lets_slow_corrections_through_better_than_fast_moves():
    # Lag relative to speed: a slow adjustment trails by much less time than a fast move.
    def lag_s(speed: float) -> float:
        shaper = MotionShaper(100)
        shaper.reset({"j": 0.0})
        targets = [speed * DT * k for k in range(1, 300)]
        return (targets[-1] - _run(shaper, targets)[-1]) / speed

    assert lag_s(3.0) < 0.4 * lag_s(120.0)


def test_pace_is_measured_on_the_raw_command():
    # Same raw series, smoothing on vs off: the pace — and therefore the strength — must
    # not depend on what the filter output did. Switch smoothing on after a fast ramp: the
    # very next step already runs at full strength.
    shaper = MotionShaper(0)
    shaper.reset({"j": 0.0})
    _run(shaper, [120.0 * DT * k for k in range(1, 60)])
    shaper.set_smoothing(100)
    assert shaper.smoothing_pct == 100
    y0 = 120.0 * DT * 59
    x = y0 + 120.0 * DT
    out = shaper.step({"j": x}, DT)["j"]
    alpha = 1 - math.exp(-DT / SMOOTHING_TAU_MAX_S)
    assert out == pytest.approx(y0 + alpha * (x - y0), rel=1e-12)
    assert PACE_TAU_S > 0  # the pace EMA's constant is a module tunable


def test_smoothing_has_no_momentum():
    # Every output lies between the previous output and this tick's target, so a
    # correction mid-move is never overshot. That is what an acceleration limit could
    # not promise, and why it was dropped.
    rng = random.Random(3)
    for smoothing in (5, 50, 100):
        shaper = MotionShaper(smoothing)
        shaper.reset({"j": 0.0})
        prev, x = 0.0, 0.0
        for _ in range(5000):
            x += rng.uniform(-3, 3) if rng.random() < 0.95 else rng.uniform(-60, 60)
            y = shaper.step({"j": x}, DT)["j"]
            assert min(prev, x) - EPS <= y <= max(prev, x) + EPS
            prev = y


def test_stop_mid_move_is_not_overshot():
    # Cruising at 100 deg/s, then the policy says "stop here": the arm never passes it.
    shaper = MotionShaper(100)
    shaper.reset({"j": 0.0})
    out = _run(shaper, [100.0 * DT * k for k in range(1, 91)])
    stop = out[-1] + 1.0  # just ahead of where the arm is now
    after = _run(shaper, [stop] * 60)
    assert max(after) <= stop + EPS
    assert all(b >= a for a, b in zip(after, after[1:], strict=False))


# --------------------------------------------------------------------------- state


def test_reset_with_a_pose_starts_there():
    shaper = MotionShaper(100)
    shaper.reset({"j": 0.0})
    _run(shaper, [90.0] * 20)
    shaper.reset({"j": 10.0})
    # The 40° jump drives the pace past PACE_REF_DEG_S at once: full time constant.
    alpha = 1 - math.exp(-DT / SMOOTHING_TAU_MAX_S)
    assert shaper.step({"j": 50.0}, DT)["j"] == pytest.approx(10.0 + alpha * 40.0)


def test_reset_without_a_pose_forgets_everything():
    shaper = MotionShaper(100)
    shaper.reset({"j": 0.0})
    shaper.reset()
    assert shaper.step({"j": 50.0}, DT) == {"j": 50.0}  # first sight: seeded there


def test_a_new_key_is_seeded_without_a_jump():
    shaper = MotionShaper(80)
    shaper.reset({"a": 0.0})
    assert shaper.step({"a": 0.0, "b": 50.0}, DT)["b"] == 50.0
    assert 50.0 < shaper.step({"a": 0.0, "b": 60.0}, DT)["b"] < 60.0
