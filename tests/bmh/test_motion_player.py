"""Unit tests for the fixed-recording replay plan (``bmh/motion/player.py``) — stdlib only.

``goto-start`` ramps monotonically to the first kept frame and then tracks the frames;
``blend-in`` starts exactly at the present pose and equals the trajectory after
``blend_s``; playback is time-indexed (a 30 Hz loop hits interpolated values of a 60 Hz
file); the kept range is respected; ``done`` after ``duration_s``; keys the robot does not
have are dropped.
"""

import pytest

from bmh.motion.file import MotionFile
from bmh.motion.player import DEFAULT_BLEND_S, DEFAULT_RAMP_S, frame_at, plan_replay, smoothstep

FPS = 60
KEYS = ("a.pos", "b.pos")


def _motion(n: int = 61, fps: int = FPS) -> MotionFile:
    """`a` = frame index, `b` = -10 * frame index: every frame's values name the frame."""
    return MotionFile(
        fps=fps, keys=KEYS, frames=tuple((float(i), -10.0 * i) for i in range(n)), recorded_at=0.0
    )


PRESENT = {"a.pos": 100.0, "b.pos": 50.0, "c.pos": 7.0}


def _monotonic(values: list[float]) -> bool:
    return all(b >= a for a, b in zip(values, values[1:], strict=False)) or all(
        b <= a for a, b in zip(values, values[1:], strict=False)
    )


def test_smoothstep_endpoints():
    assert smoothstep(-1.0) == 0.0 and smoothstep(0.0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(1.0) == 1.0 and smoothstep(2.0) == 1.0


def test_frame_at_interpolates_and_clamps():
    frames = ((0.0, 0.0), (10.0, -10.0), (20.0, -20.0))
    assert frame_at(frames, 10, -1.0) == (0.0, 0.0)
    assert frame_at(frames, 10, 0.0) == (0.0, 0.0)
    assert frame_at(frames, 10, 0.05) == pytest.approx((5.0, -5.0))
    assert frame_at(frames, 10, 0.1) == (10.0, -10.0)
    assert frame_at(frames, 10, 0.175) == pytest.approx((17.5, -17.5))
    assert frame_at(frames, 10, 0.2) == (20.0, -20.0)
    assert frame_at(frames, 10, 9.0) == (20.0, -20.0)
    assert frame_at(((3.0,),), 10, 5.0) == (3.0,)
    with pytest.raises(ValueError):
        frame_at((), 10, 0.0)


def test_goto_start_ramps_monotonically_then_tracks_the_frames():
    plan = plan_replay(PRESENT, _motion(), 0, 61, "goto-start", ramp_s=1.0)
    ramp = [plan.pose_at(i / 100) for i in range(101)]
    assert ramp[0] == {"a.pos": 100.0, "b.pos": 50.0}
    for key in KEYS:
        assert _monotonic([pose[key] for pose in ramp])
    assert plan.pose_at(1.0) == {"a.pos": 0.0, "b.pos": -0.0}  # frames[start] at ramp_s
    assert plan.phase_at(0.0) == plan.phase_at(0.99) == "ramping"
    # After the ramp: the recording itself, at its own time.
    assert plan.pose_at(1.0 + 10 / FPS) == pytest.approx({"a.pos": 10.0, "b.pos": -100.0})
    assert plan.phase_at(1.0) == plan.phase_at(1.5) == "playing"
    assert plan.duration_s == pytest.approx(2.0)
    assert plan.phase_at(2.0) == "done"
    assert plan.pose_at(2.0) == plan.pose_at(99.0) == {"a.pos": 60.0, "b.pos": -600.0}


def test_blend_in_starts_at_the_present_pose_and_joins_the_trajectory():
    plan = plan_replay(PRESENT, _motion(), 0, 61, "blend-in", blend_s=0.5)
    assert plan.pose_at(0.0) == {"a.pos": 100.0, "b.pos": 50.0}
    assert plan.phase_at(0.0) == "playing"
    # Halfway through the blend the offset is halved (smoothstep(0.5) == 0.5).
    t = 0.25
    on_track = frame_at(plan.frames, FPS, t)
    assert plan.pose_at(t) == pytest.approx({"a.pos": on_track[0] + 50.0, "b.pos": on_track[1] + 25.0})
    # From blend_s on, exactly the recording.
    for t in (0.5, 0.6, 0.9):
        expected = frame_at(plan.frames, FPS, t)
        assert plan.pose_at(t) == {"a.pos": expected[0], "b.pos": expected[1]}
    assert plan.duration_s == pytest.approx(1.0)
    assert plan.phase_at(1.0) == "done"


def test_blend_longer_than_the_recording_ends_after_the_blend():
    plan = plan_replay(PRESENT, _motion(n=7), 0, 7, "blend-in", blend_s=1.0)
    assert plan.play_s == pytest.approx(0.1)
    assert plan.duration_s == 1.0
    assert plan.phase_at(0.5) == "playing"
    assert plan.pose_at(1.0) == {"a.pos": 6.0, "b.pos": -60.0}


def test_playback_is_time_indexed_not_tick_indexed():
    # A 30 Hz loop over a 60 Hz file: every tick lands on every other frame...
    plan = plan_replay(PRESENT, _motion(), 0, 61, "blend-in", blend_s=0.0)
    for tick in range(31):
        assert plan.pose_at(tick / 30) == pytest.approx({"a.pos": 2.0 * tick, "b.pos": -20.0 * tick})
    # ...and a tick between two frames gets the interpolated value.
    assert plan.pose_at(1.25 / FPS) == pytest.approx({"a.pos": 1.25, "b.pos": -12.5})


def test_range_is_respected():
    plan = plan_replay(PRESENT, _motion(), 10, 20, "goto-start", ramp_s=0.5)
    assert len(plan.frames) == 10
    assert plan.pose_at(0.5) == {"a.pos": 10.0, "b.pos": -100.0}  # frames[start]
    seen = [plan.pose_at(t / 1000)["a.pos"] for t in range(0, 5000)]
    assert min(v for v in seen[500:]) == 10.0 and max(seen) == 100.0 and max(seen[500:]) == 19.0
    assert plan.duration_s == pytest.approx(0.5 + 9 / FPS)


@pytest.mark.parametrize(("start", "end"), [(-1, 5), (5, 5), (6, 5), (0, 62), (61, 62)])
def test_range_outside_the_recording_is_rejected(start, end):
    with pytest.raises(ValueError, match="range"):
        plan_replay(PRESENT, _motion(), start, end, "goto-start")


def test_unknown_mode_and_negative_durations_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        plan_replay(PRESENT, _motion(), 0, 5, "fast")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plan_replay(PRESENT, _motion(), 0, 5, "goto-start", ramp_s=-1.0)


def test_keys_missing_from_the_present_pose_are_dropped():
    plan = plan_replay({"a.pos": 1.0}, _motion(), 0, 61, "goto-start")
    assert plan.keys == ("a.pos",) and plan.dropped_keys == ("b.pos",)
    assert set(plan.pose_at(0.0)) == {"a.pos"}
    with pytest.raises(ValueError, match="no joint"):
        plan_replay({"z.pos": 1.0}, _motion(), 0, 61, "goto-start")


def test_present_keys_the_recording_lacks_are_left_out():
    plan = plan_replay(PRESENT, _motion(), 0, 61, "goto-start")
    assert set(plan.pose_at(1.0)) == set(KEYS)  # `c.pos` is not commanded


def test_defaults():
    assert (DEFAULT_RAMP_S, DEFAULT_BLEND_S) == (2.5, 1.0)
    goto = plan_replay(PRESENT, _motion(), 0, 61, "goto-start")
    blend = plan_replay(PRESENT, _motion(), 0, 61, "blend-in")
    assert (goto.ramp_s, goto.blend_s, blend.ramp_s, blend.blend_s) == (2.5, 0.0, 0.0, 1.0)
