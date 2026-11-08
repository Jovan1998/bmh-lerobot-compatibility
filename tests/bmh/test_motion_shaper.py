"""Unit tests for ``bmh.policy_client.motion`` — the look-ahead and the speed-gated
smoothing the policy client applies between the chunk and the servos.

Pure stdlib module, no optional deps. Synthetic series at 30 Hz: both stages at 0 must
return the input bit for bit; smoothing must follow a first-order step response whose
time constant grows with the raw command's pace, must never carry the arm past a target
(no momentum), and must ease in when its slider is lowered instead of jumping; the
look-ahead must keep a ramp on the ramp (no lag), damp jitter, shrink symmetrically and
stay within its window. Run with ``uv run --no-sync pytest tests/bmh/test_motion_shaper.py -q``.
"""

import math
import random

import pytest

from bmh.policy_client.motion import (
    LOOKAHEAD_MAX_FRAMES,
    PACE_FLOOR,
    PACE_REF_DEG_S,
    PACE_TAU_S,
    SMOOTHING_TAU_MAX_S,
    LookaheadSmoother,
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


# --------------------------------------------------------------------------- lowering the slider


def _trailing(shaper: MotionShaper, speed: float = 60.0, seconds: float = 1.5) -> tuple[float, float]:
    """Cruise at `speed` deg/s for `seconds` (the arm trails behind).

    Returns the last command and the last output.
    """
    shaper.reset({"j": 0.0})
    x = y = 0.0
    for _ in range(round(seconds / DT)):
        x += speed * DT
        y = shaper.step({"j": x}, DT)["j"]
    return x, y


@pytest.mark.parametrize(("start", "bound"), [(100, 5.5), (30, 3.5), (50, 4.5)])
def test_switching_off_mid_move_eases_in_instead_of_jumping(start, bound):
    # Before: the whole lag (29 deg from 100 %, 10 deg from 30 %) went out in one tick.
    shaper = MotionShaper(start)
    x, y = _trailing(shaper)
    assert x - y > 7.0  # trailing far behind (8 deg at 30 %, 27 at 100 %)
    shaper.set_smoothing(0)
    assert shaper.easing
    steps, exact_after = [], None
    for k in range(1, 61):
        x += 60.0 * DT
        out = shaper.step({"j": x}, DT)
        steps.append(out["j"] - y)
        y = out["j"]
        assert y <= x + EPS  # never past the command
        if exact_after is None and out == {"j": x}:
            exact_after = k
    assert max(steps) < bound  # a normal tick moves 2 deg
    assert exact_after is not None and exact_after * DT < 1.0
    assert not shaper.easing


def test_once_eased_in_off_is_bit_for_bit_again():
    rng = random.Random(3)
    shaper = MotionShaper(80)
    _trailing(shaper)
    shaper.set_smoothing(0)
    _run(shaper, [rng.uniform(-90, 90) for _ in range(60)])  # 2 s: long done easing
    for _ in range(200):
        target = {"j": rng.uniform(-180, 180)}
        out = shaper.step(target, DT)
        assert out == target and out["j"] is target["j"]


def test_lowering_but_not_off_eases_in_too():
    shaper = MotionShaper(100)
    x, y = _trailing(shaper)
    shaper.set_smoothing(10)
    first = shaper.step({"j": x + 60.0 * DT}, DT)["j"]
    assert first - y < 3.0  # was 14 deg in one tick
    _run(shaper, [x] * 60)
    assert not shaper.easing  # settled at the new, lower value


def test_raising_applies_at_once():
    shaper = MotionShaper(10)
    shaper.reset({"j": 0.0})
    shaper.set_smoothing(100)
    assert not shaper.easing
    alpha = 1 - math.exp(-DT / SMOOTHING_TAU_MAX_S)
    # A 50° step drives the pace past PACE_REF_DEG_S at once: full time constant.
    assert shaper.step({"j": 50.0}, DT)["j"] == pytest.approx(alpha * 50.0)


def test_reset_ends_the_ease_in():
    shaper = MotionShaper(100)
    _trailing(shaper)
    shaper.set_smoothing(0)
    shaper.reset({"j": 5.0})
    assert not shaper.easing
    assert shaper.step({"j": 7.0}, DT) == {"j": 7.0}


# --------------------------------------------------------------------------- look-ahead


def _play(smoother: LookaheadSmoother, path: list[float], key: str = "j") -> list[float]:
    """Play `path` as one chunk from its first action: each step sees the rest as planned."""
    steps = [{key: v} for v in path]
    return [smoother.step(steps[i], steps[i + 1 : i + 1 + smoother.frames])[key] for i in range(len(steps))]


def test_lookahead_off_returns_the_input_bit_for_bit():
    rng = random.Random(11)
    smoother = LookaheadSmoother(0)
    for _ in range(500):
        target = {"a": rng.uniform(-180, 180), "b": rng.random() * 100}
        upcoming = [{"a": rng.uniform(-180, 180), "b": 1.0} for _ in range(5)]
        out = smoother.step(target, upcoming)
        assert out == target and out is not target
        assert all(out[k] is target[k] for k in target)


@pytest.mark.parametrize("frames", [1, 2, 3, 6, LOOKAHEAD_MAX_FRAMES])
def test_lookahead_keeps_a_ramp_on_the_ramp(frames):
    # Centred, so no lag: a steady move comes out exactly where the policy put it.
    path = [3.0 * k - 40.0 for k in range(40)]
    out = _play(LookaheadSmoother(frames), path)
    assert out == pytest.approx(path, abs=1e-9)


@pytest.mark.parametrize(("frames", "left"), [(1, 0.6), (2, 0.05), (3, 0.03)])
def test_lookahead_removes_tick_to_tick_jitter(frames, left):
    path = [5.0 + (0.5 if k % 2 else -0.5) for k in range(40)]
    out = _play(LookaheadSmoother(frames), path)
    middle = out[frames : len(out) - frames]  # full window
    assert max(abs(v - 5.0) for v in middle) < left * 0.5


def test_lookahead_shrinks_symmetrically_where_the_chunk_runs_out():
    smoother = LookaheadSmoother(3)
    _play(smoother, [0.0, 1.0, 2.0, 3.0])
    # Only one planned action left: 1 back, the action, 1 ahead — still centred.
    out = smoother.step({"j": 4.0}, [{"j": 50.0}])["j"]
    weights = [math.exp(-(k * k) / (2 * 1.5**2)) for k in (-1, 0, 1)]
    assert out == pytest.approx((3.0 * weights[0] + 4.0 * weights[1] + 50.0 * weights[2]) / sum(weights))
    # Nothing planned (late chunk): the action as is.
    assert smoother.step({"j": 7.0}, []) == {"j": 7.0}


def test_lookahead_output_stays_within_its_window():
    # All weights are positive: no overshoot, whatever the commands do.
    rng = random.Random(5)
    path = [rng.uniform(-50, 50) for _ in range(300)]
    for frames in (1, 3, 6):
        out = _play(LookaheadSmoother(frames), path)
        for i, v in enumerate(out):
            window = path[max(0, i - frames) : i + frames + 1]
            assert min(window) - EPS <= v <= max(window) + EPS


def test_lookahead_clear_restarts_from_the_raw_action():
    smoother = LookaheadSmoother(3)
    _play(smoother, [30.0] * 10)
    smoother.clear()
    assert smoother.step({"j": 0.0}, [{"j": 0.0}] * 3) == {"j": 0.0}  # not mixed with the 30s
    assert smoother.step({"j": 0.0}, [{"j": 0.0}] * 3)["j"] == pytest.approx(0.0)


def test_lookahead_other_joints_restart_the_history():
    smoother = LookaheadSmoother(2)
    _play(smoother, [30.0] * 5, key="a")
    assert smoother.step({"b": 1.0}, [{"b": 9.0}] * 2) == {"b": 1.0}


def test_lookahead_future_ends_at_an_entry_without_the_joint():
    smoother = LookaheadSmoother(3)
    _play(smoother, [0.0] * 5)
    # The second planned entry lacks "j": only one planned action is usable, so h = 1.
    out = smoother.step({"j": 0.0}, [{"j": 3.0}, {"x": 99.0}, {"j": 99.0}])["j"]
    assert 0.0 < out < 3.0


def test_lookahead_covers_exactly_the_target_keys():
    smoother = LookaheadSmoother(2)
    _play(smoother, [1.0] * 4)
    out = smoother.step({"j": 1.0}, [{"j": 1.0, "feedback_done": 0.5}] * 2)
    assert set(out) == {"j"}


def test_lookahead_frames_change_live_and_keep_the_history():
    smoother = LookaheadSmoother(0)
    path = [5.0 + (0.5 if k % 2 else -0.5) for k in range(10)]
    for i in range(6):
        smoother.step({"j": path[i]}, [{"j": v} for v in path[i + 1 : i + 4]])
    smoother.set_frames(3)
    assert smoother.frames == 3
    # The history was kept while off: the very next step uses a full window.
    out = smoother.step({"j": path[6]}, [{"j": v} for v in path[7:10]])["j"]
    assert abs(out - 5.0) < 0.03
