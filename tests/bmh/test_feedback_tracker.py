"""Unit tests for the policy feedback tracker - no numpy, no ZMQ, no hardware.

Covers ``FeedbackTracker``: hysteresis, the N-consecutive-tick debounce, the
full-snapshot-on-change contract (first sight announces the keys all-false, a key-set
change is a change) and ``reset``. Pure stdlib module, so this runs anywhere.
Run with ``uv run --no-sync pytest tests/bmh/test_feedback_tracker.py -q``.
"""

import pytest

from bmh.groot_client.feedback import FeedbackTracker


def _tracker(debounce: int = 3) -> FeedbackTracker:
    return FeedbackTracker(on_threshold=0.6, off_threshold=0.4, debounce_ticks=debounce)


def _feed(tracker: FeedbackTracker, key: str, values: list[float]) -> list[dict[str, bool] | None]:
    return [tracker.update({key: v}) for v in values]


# --------------------------------------------------------------------------- snapshots


def test_first_sight_announces_all_keys_false_then_stays_quiet():
    tracker = _tracker()
    assert tracker.update({"done": 0.0, "object_grasped": 0.9}) == {"done": False, "object_grasped": False}
    assert tracker.update({"done": 0.0, "object_grasped": 0.0}) is None
    assert tracker.update({"done": 0.1, "object_grasped": 0.2}) is None


def test_no_feedbacks_never_reports_anything():
    tracker = _tracker()
    assert [tracker.update({}) for _ in range(5)] == [None] * 5
    assert tracker.reset() is False


def test_flip_reports_the_full_snapshot_not_a_delta():
    tracker = _tracker(debounce=2)
    tracker.update({"a": 0.0, "b": 0.0})
    assert tracker.update({"a": 0.9, "b": 0.0}) is None
    assert tracker.update({"a": 0.9, "b": 0.0}) == {"a": True, "b": False}
    assert tracker.update({"a": 0.9, "b": 0.9}) is None
    assert tracker.update({"a": 0.9, "b": 0.9}) == {"a": True, "b": True}


def test_key_set_change_is_a_change_and_keeps_surviving_flags():
    tracker = _tracker(debounce=1)
    assert tracker.update({"a": 0.9}) == {"a": True}  # debounce 1: the first tick already counts
    assert tracker.update({"a": 0.9, "b": 0.0}) == {"a": True, "b": False}
    assert tracker.update({"b": 0.0}) == {"b": False}
    assert tracker.update({"b": 0.0}) is None


def test_snapshot_is_a_copy():
    tracker = _tracker(debounce=1)
    snapshot = tracker.update({"a": 0.0})
    snapshot["a"] = True
    assert tracker.update({"a": 0.0}) is None  # the tracker's own state was not touched


# --------------------------------------------------------------------------- debounce


def test_switches_on_only_after_n_consecutive_ticks():
    tracker = _tracker(debounce=3)
    tracker.update({"done": 0.0})
    assert _feed(tracker, "done", [0.9, 0.9]) == [None, None]
    assert tracker.update({"done": 0.9}) == {"done": True}
    assert _feed(tracker, "done", [0.9] * 10) == [None] * 10  # emitted once, on the change


def test_a_single_tick_back_restarts_the_count():
    tracker = _tracker(debounce=3)
    tracker.update({"done": 0.0})
    assert _feed(tracker, "done", [0.9, 0.9, 0.5, 0.9, 0.9]) == [None] * 5
    assert tracker.update({"done": 0.9}) == {"done": True}


def test_thresholds_are_inclusive():
    tracker = _tracker(debounce=1)
    tracker.update({"done": 0.0})
    assert tracker.update({"done": 0.59}) is None
    assert tracker.update({"done": 0.6}) == {"done": True}
    assert tracker.update({"done": 0.41}) is None
    assert tracker.update({"done": 0.4}) == {"done": False}


# --------------------------------------------------------------------------- hysteresis


def test_value_inside_the_band_keeps_the_current_flag():
    tracker = _tracker(debounce=2)
    tracker.update({"done": 0.0})
    assert _feed(tracker, "done", [0.5] * 6) == [None] * 6  # off stays off
    assert _feed(tracker, "done", [0.9, 0.9]) == [None, {"done": True}]
    assert _feed(tracker, "done", [0.5] * 6) == [None] * 6  # on stays on
    assert _feed(tracker, "done", [0.1, 0.5, 0.1]) == [None] * 3  # the band restarts the off-count
    assert tracker.update({"done": 0.1}) == {"done": False}


def test_noise_around_one_threshold_does_not_flap():
    tracker = _tracker(debounce=1)
    tracker.update({"done": 0.0})
    reports = _feed(tracker, "done", [0.61, 0.59, 0.62, 0.58, 0.61, 0.57])
    assert reports == [{"done": True}, None, None, None, None, None]


def test_nan_never_counts():
    tracker = _tracker(debounce=2)
    tracker.update({"done": 0.0})
    assert _feed(tracker, "done", [0.9, float("nan"), 0.9]) == [None] * 3
    assert tracker.update({"done": 0.9}) == {"done": True}


# --------------------------------------------------------------------------- reset


def test_reset_reports_whether_the_app_was_shown_anything():
    tracker = _tracker(debounce=1)
    assert tracker.reset() is False  # nothing seen yet
    tracker.update({"done": 0.0})
    assert tracker.reset() is True  # an all-false snapshot is still something on screen
    assert tracker.reset() is False  # ...and only once


def test_reset_forgets_flags_streaks_and_keys():
    tracker = _tracker(debounce=3)
    tracker.update({"done": 0.0})
    _feed(tracker, "done", [0.9, 0.9])  # one tick short of switching on
    assert tracker.reset() is True
    # First sight again, and the two ticks from before the reset are gone.
    assert _feed(tracker, "done", [0.9, 0.9]) == [{"done": False}, None]
    assert tracker.update({"done": 0.9}) == {"done": True}
    assert tracker.reset() is True
    assert tracker.update({"done": 0.9}) == {"done": False}


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize(
    ("on", "off", "debounce"),
    [(0.4, 0.6, 5), (0.6, 0.4, 0), (0.6, 0.4, -1)],
)
def test_rejects_flapping_thresholds_and_empty_debounce(on: float, off: float, debounce: int):
    with pytest.raises(ValueError):
        FeedbackTracker(on_threshold=on, off_threshold=off, debounce_ticks=debounce)


def test_equal_thresholds_are_allowed():
    tracker = FeedbackTracker(on_threshold=0.5, off_threshold=0.5, debounce_ticks=1)
    assert tracker.update({"done": 0.5}) == {"done": True}
