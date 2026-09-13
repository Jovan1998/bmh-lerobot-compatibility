"""Unit tests for the BMH-101 teleop group lock - no ZMQ, no hardware.

Covers ``ActionGroupLock`` (freeze + smoothstep unlock blend) and ``LockFileWatcher``
(stat-based change detection of the controller-app's JSON file). Needs ``draccus``
because importing the teleoperator package registers draccus configs. Run with
``uv run --no-sync pytest tests/bmh/test_group_lock.py -q``.
"""

import json
import os
from itertools import pairwise

import pytest

pytest.importorskip("draccus")

from lerobot.teleoperators.bi_so_network_leader.group_lock import (  # noqa: E402
    LOCK_GROUPS,
    ActionGroupLock,
    LockFileWatcher,
    bimanual_lock_groups,
    smoothstep,
)
from lerobot.teleoperators.so_network_leader.so_network_leader import (  # noqa: E402
    ARM_MOTOR_NAMES,
    HEAD_MOTOR_NAMES,
)

BLEND_S = 0.8
GROUPS = bimanual_lock_groups(left_with_head=True)
LEFT, RIGHT, HEAD = GROUPS["left"], GROUPS["right"], GROUPS["head"]
ALL_UNLOCKED = {"left": False, "right": False, "head": False}


def _action(left: float = 0.0, right: float = 0.0, head: float = 0.0) -> dict[str, float]:
    """Bimanual action dict with every joint of a group set to the same value."""
    action = {f"left_{m}.pos": left for m in ARM_MOTOR_NAMES}
    action.update({f"right_{m}.pos": right for m in ARM_MOTOR_NAMES})
    action.update({f"left_{m}.pos": head for m in HEAD_MOTOR_NAMES})
    return action


def _lock(blend_s: float = BLEND_S) -> ActionGroupLock:
    return ActionGroupLock(bimanual_lock_groups(left_with_head=True), blend_s=blend_s)


def _all(out: dict[str, float], keys: list[str], value: float) -> bool:
    return all(out[k] == pytest.approx(value) for k in keys)


# --------------------------------------------------------------------------- groups


def test_groups_cover_expected_keys():
    assert set(GROUPS) == set(LOCK_GROUPS)
    assert len(LEFT) == 7 and "left_gripper.pos" in LEFT
    assert len(RIGHT) == 7 and "right_gripper.pos" in RIGHT
    assert HEAD == ["left_head_pan.pos", "left_head_tilt.pos"]
    assert not (set(LEFT) & set(HEAD))
    assert bimanual_lock_groups(left_with_head=False)["head"] == []


def test_smoothstep_endpoints():
    assert smoothstep(-1.0) == 0.0 and smoothstep(0.0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(1.0) == 1.0 and smoothstep(2.0) == 1.0


# --------------------------------------------------------------------------- hold


def test_passthrough_when_nothing_locked():
    lock = _lock()
    action = _action(left=1.0, right=2.0, head=3.0)
    out = lock.apply(action, now=0.0)
    assert out == action and out is not action
    assert lock.locks == ALL_UNLOCKED


def test_locked_group_holds_while_leader_moves():
    lock = _lock()
    lock.apply(_action(left=10.0, right=10.0, head=10.0), now=0.0)
    lock.set_locks({"left": True})
    out = lock.apply(_action(left=50.0, right=50.0, head=50.0), now=0.1)
    assert _all(out, LEFT, 10.0)
    assert _all(out, RIGHT, 50.0)
    assert _all(out, HEAD, 50.0)
    assert lock.modes == {"left": "hold", "right": "track", "head": "track"}


def test_lock_on_first_tick_holds_live_leader_values():
    lock = _lock()
    lock.set_locks({"right": True})
    out = lock.apply(_action(right=7.0), now=0.0)
    assert _all(out, RIGHT, 7.0)
    out = lock.apply(_action(right=99.0), now=0.1)
    assert _all(out, RIGHT, 7.0)


def test_head_and_left_are_independent_despite_shared_prefix():
    lock = _lock()
    lock.apply(_action(left=1.0, head=1.0), now=0.0)

    lock.set_locks({"head": True})
    out = lock.apply(_action(left=2.0, head=2.0), now=0.1)
    assert _all(out, LEFT, 2.0)
    assert _all(out, HEAD, 1.0)

    lock.set_locks({"left": True})  # head stays locked
    out = lock.apply(_action(left=3.0, head=3.0), now=0.2)
    assert _all(out, LEFT, 2.0)
    assert _all(out, HEAD, 1.0)
    assert _all(out, RIGHT, 0.0)


def test_gripper_is_part_of_the_arm_group():
    lock = _lock()
    lock.set_locks({"left": True})
    lock.apply(_action(left=40.0), now=0.0)
    out = lock.apply(_action(left=0.0), now=0.1)
    assert out["left_gripper.pos"] == 40.0


def test_keys_missing_from_action_are_skipped():
    # Leader host streaming without head keys + head lock: no KeyError, nothing changes.
    lock = _lock()
    lock.set_locks({"head": True})
    action = dict.fromkeys(LEFT + RIGHT, 1.0)
    assert lock.apply(action, now=0.0) == action


def test_unknown_group_is_ignored():
    lock = _lock()
    lock.set_locks({"torso": True, "left": True})
    assert lock.locks == {"left": True, "right": False, "head": False}


# --------------------------------------------------------------------------- blend


def _held_then_unlocked(held: float) -> ActionGroupLock:
    lock = _lock()
    lock.set_locks({"left": True})
    lock.apply(_action(left=held), now=0.0)
    lock.set_locks({"left": False})
    return lock


def test_unlock_blends_from_held_pose_to_leader():
    lock = _held_then_unlocked(10.0)

    # The tick that observes the unlock still outputs exactly the held pose (no jump).
    out = lock.apply(_action(left=30.0), now=1.0)
    assert _all(out, LEFT, 10.0)
    assert lock.modes["left"] == "blend"
    # Halfway: smoothstep(0.5) == 0.5.
    out = lock.apply(_action(left=30.0), now=1.0 + BLEND_S / 2)
    assert _all(out, LEFT, 20.0)
    # End of blend: exactly the leader, then pure passthrough.
    out = lock.apply(_action(left=30.0), now=1.0 + BLEND_S)
    assert _all(out, LEFT, 30.0)
    assert lock.modes["left"] == "track"
    out = lock.apply(_action(left=31.0), now=5.0)
    assert _all(out, LEFT, 31.0)


def test_blend_is_monotonic_and_eases_in_and_out():
    lock = _held_then_unlocked(0.0)
    steps = 40
    samples = [
        lock.apply(_action(left=100.0), now=i * BLEND_S / steps)["left_shoulder_pan.pos"]
        for i in range(steps + 1)
    ]
    assert samples[0] == 0.0 and samples[-1] == pytest.approx(100.0)
    assert all(b >= a for a, b in pairwise(samples))
    first = samples[1] - samples[0]
    middle = samples[steps // 2 + 1] - samples[steps // 2]
    last = samples[-1] - samples[-2]
    assert first < middle and last < middle


def test_blend_tracks_a_leader_that_keeps_moving():
    lock = _held_then_unlocked(10.0)
    lock.apply(_action(left=30.0), now=0.0)
    out = lock.apply(_action(left=50.0), now=BLEND_S / 2)  # leader moved during the blend
    assert _all(out, LEFT, 30.0)  # 10 + 0.5 * (50 - 10)


def test_lock_mid_blend_holds_the_blended_pose():
    lock = _held_then_unlocked(10.0)
    lock.apply(_action(left=30.0), now=0.0)
    out = lock.apply(_action(left=30.0), now=BLEND_S / 2)
    assert _all(out, LEFT, 20.0)

    lock.set_locks({"left": True})
    out = lock.apply(_action(left=80.0), now=BLEND_S / 2 + 0.01)
    assert _all(out, LEFT, 20.0)
    out = lock.apply(_action(left=-80.0), now=10.0)
    assert _all(out, LEFT, 20.0)


def test_blend_only_touches_the_unlocking_group():
    lock = _held_then_unlocked(10.0)
    lock.apply(_action(left=30.0, right=30.0, head=30.0), now=0.0)
    out = lock.apply(_action(left=30.0, right=31.0, head=32.0), now=BLEND_S / 2)
    assert _all(out, LEFT, 20.0)
    assert _all(out, RIGHT, 31.0)
    assert _all(out, HEAD, 32.0)


def test_zero_blend_duration_snaps_to_leader():
    lock = _lock(blend_s=0.0)
    lock.set_locks({"right": True})
    lock.apply(_action(right=1.0), now=0.0)
    lock.set_locks({"right": False})
    out = lock.apply(_action(right=9.0), now=0.0)
    assert _all(out, RIGHT, 9.0)
    assert lock.modes["right"] == "track"


def test_negative_blend_duration_rejected():
    with pytest.raises(ValueError):
        _lock(blend_s=-0.1)


# --------------------------------------------------------------------------- watcher


def _write(path, data, mtime_ns: int) -> None:
    path.write_text(json.dumps(data))
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_watcher_missing_file_reports_all_unlocked_once(tmp_path):
    watcher = LockFileWatcher(tmp_path / "locks.json")
    assert watcher.poll() == ALL_UNLOCKED
    assert watcher.poll() is None


def test_watcher_reads_only_when_file_changes(tmp_path):
    path = tmp_path / "locks.json"
    watcher = LockFileWatcher(path)
    _write(path, {"left": True, "right": False, "head": True}, 1_000)
    assert watcher.poll() == {"left": True, "right": False, "head": True}
    assert watcher.poll() is None
    _write(path, {"left": False, "right": False, "head": True}, 2_000)
    assert watcher.poll() == {"left": False, "right": False, "head": True}
    assert watcher.poll() is None


def test_watcher_missing_keys_default_unlocked_and_unknown_keys_ignored(tmp_path):
    path = tmp_path / "locks.json"
    watcher = LockFileWatcher(path)
    _write(path, {"left": True, "torso": True}, 1_000)
    assert watcher.poll() == {"left": True, "right": False, "head": False}


def test_watcher_malformed_file_keeps_previous_state_and_warns_once(tmp_path, caplog):
    path = tmp_path / "locks.json"
    watcher = LockFileWatcher(path)
    _write(path, {"left": True}, 1_000)
    assert watcher.poll()["left"] is True

    path.write_text("{not json")
    os.utime(path, ns=(2_000, 2_000))
    with caplog.at_level("WARNING"):
        assert watcher.poll() is None
        assert watcher.poll() is None
    assert caplog.text.lower().count("lock file") == 1

    _write(path, {"left": 1}, 3_000)  # wrong type
    with caplog.at_level("WARNING"):
        assert watcher.poll() is None


def test_watcher_deleted_file_clears_locks(tmp_path):
    path = tmp_path / "locks.json"
    watcher = LockFileWatcher(path)
    _write(path, {"left": True}, 1_000)
    assert watcher.poll()["left"] is True
    path.unlink()
    assert watcher.poll() == ALL_UNLOCKED
    assert watcher.poll() is None
