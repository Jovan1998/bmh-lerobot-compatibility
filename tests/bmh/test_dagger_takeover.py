"""`bmh-dagger` take-over: alignment per joint kind, the auto-start gate and the blend into teleop."""

import pytest

from bmh.dagger.takeover import AlignGate, TakeoverBlend, alignment

HOLD = {"left_shoulder_pan.pos": 10.0, "left_gripper.pos": 40.0, "right_wrist_roll.pos": -5.0}


def _leader(**delta: float) -> dict[str, float]:
    return {key: value + delta.get(key.split(".")[0], 0.0) for key, value in HOLD.items()}


# --------------------------------------------------------------------------- alignment


def test_diff_is_leader_minus_hold():
    state = alignment(_leader(left_shoulder_pan=-3.0, left_gripper=2.0), HOLD, 4.0, 5.0)
    assert state.diff == {"left_shoulder_pan.pos": -3.0, "left_gripper.pos": 2.0, "right_wrist_roll.pos": 0.0}
    assert state.aligned
    assert state.max_abs == 3.0


def test_arm_joints_and_grippers_have_their_own_tolerance():
    # 4.5 is outside the arm tolerance (4) but inside the gripper's (5).
    assert not alignment(_leader(left_shoulder_pan=4.5), HOLD, 4.0, 5.0).aligned
    assert alignment(_leader(left_gripper=4.5), HOLD, 4.0, 5.0).aligned
    assert not alignment(_leader(left_gripper=-5.5), HOLD, 4.0, 5.0).aligned
    # At the tolerance counts as aligned.
    assert alignment(_leader(right_wrist_roll=4.0), HOLD, 4.0, 5.0).aligned


def test_only_common_keys_are_compared_and_none_means_not_aligned():
    state = alignment({"left_shoulder_pan.pos": 10.0, "extra.pos": 99.0}, HOLD, 4.0, 5.0)
    assert state.diff == {"left_shoulder_pan.pos": 0.0}
    assert state.aligned
    empty = alignment({"other.pos": 1.0}, HOLD, 4.0, 5.0)
    assert empty.diff == {} and not empty.aligned and empty.max_abs == 0.0


# --------------------------------------------------------------------------- gate


def test_gate_needs_an_unbroken_hold():
    gate = AlignGate(0.4)
    assert not gate.update(True, 10.0)
    assert not gate.update(True, 10.3)
    assert not gate.update(False, 10.35)  # the gap restarts the count
    assert not gate.update(True, 10.4)
    assert not gate.update(True, 10.79)
    assert gate.update(True, 10.8)
    assert gate.update(True, 11.0)


def test_gate_reset():
    gate = AlignGate(0.4)
    gate.update(True, 0.0)
    gate.reset()
    assert not gate.update(True, 0.5)
    assert gate.update(True, 0.9)


# --------------------------------------------------------------------------- blend


def test_blend_starts_exactly_at_the_hold_pose():
    leader = _leader(left_shoulder_pan=3.7, left_gripper=-1.3, right_wrist_roll=0.9)
    blend = TakeoverBlend(HOLD, leader, blend_s=1.0)
    assert blend.apply(leader, 0.0) == HOLD  # bit for bit, no tolerance
    assert blend.apply(leader, -0.1) == HOLD


def test_blend_ends_exactly_on_the_leader():
    leader = _leader(left_shoulder_pan=3.7)
    blend = TakeoverBlend(HOLD, leader, blend_s=1.0)
    later = {**leader, "left_gripper.pos": 71.3}
    assert blend.apply(later, 1.0) == later
    assert blend.apply(later, 5.0) == later
    assert blend.apply(later, 5.0) is not later


def test_blend_is_monotonic_for_a_still_leader():
    leader = _leader(left_shoulder_pan=3.0)
    blend = TakeoverBlend(HOLD, leader, blend_s=1.0)
    values = [blend.apply(leader, t / 20)["left_shoulder_pan.pos"] for t in range(21)]
    assert values[0] == 10.0 and values[-1] == 13.0
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    assert values[10] == pytest.approx(11.5)  # smoothstep(0.5) = 0.5


def test_leader_motion_passes_through_on_top_of_the_fading_offset():
    leader = _leader(left_shoulder_pan=2.0)
    blend = TakeoverBlend(HOLD, leader, blend_s=1.0)
    moved = {**leader, "left_shoulder_pan.pos": leader["left_shoulder_pan.pos"] + 5.0}
    # At t = 0 the offset is whole: hold + what the leader moved.
    assert blend.apply(moved, 0.0)["left_shoulder_pan.pos"] == pytest.approx(15.0)
    # Halfway, half of the 2-unit offset is gone: leader - 1.
    assert blend.apply(moved, 0.5)["left_shoulder_pan.pos"] == pytest.approx(16.0)


def test_duration_grows_with_the_gap():
    near = TakeoverBlend(HOLD, _leader(left_shoulder_pan=3.0), blend_s=1.0)
    far = TakeoverBlend(HOLD, _leader(left_shoulder_pan=90.0), blend_s=1.0)
    assert near.duration_s == 1.0
    assert far.duration_s == pytest.approx(2.0)  # 90 / 45 per second
    leader = _leader(left_shoulder_pan=90.0)
    assert far.apply(leader, 1.0)["left_shoulder_pan.pos"] == pytest.approx(55.0)  # halfway
    assert far.apply(leader, 2.0) == leader


def test_keys_the_hold_pose_lacks_pass_through():
    leader = {**_leader(left_shoulder_pan=2.0), "left_head_pan.pos": 33.0}
    blend = TakeoverBlend(HOLD, leader, blend_s=1.0)
    out = blend.apply(leader, 0.0)
    assert out["left_head_pan.pos"] == 33.0
    assert set(out) == set(leader)
