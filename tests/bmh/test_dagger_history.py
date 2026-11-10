"""`bmh-dagger` rewind history: bounded, addressed by a cursor, truncated when the motion forks."""

import pytest

from bmh.dagger.history import ActionHistory


def _filled(n: int, max_frames: int = 100) -> ActionHistory:
    history = ActionHistory(max_frames)
    for i in range(n):
        history.push({"j": float(i)})
    return history


def test_cursor_counts_back_from_the_newest_frame():
    history = _filled(5)
    assert len(history) == 5
    assert history.at(0) == {"j": 4.0}
    assert history.at(4) == {"j": 0.0}


def test_bound_drops_the_oldest_frames():
    history = _filled(10, max_frames=4)
    assert len(history) == 4
    assert [history.at(c)["j"] for c in range(4)] == [9.0, 8.0, 7.0, 6.0]


def test_at_rejects_a_cursor_outside_the_history():
    history = _filled(3)
    for cursor in (-1, 3):
        with pytest.raises(IndexError):
            history.at(cursor)
    with pytest.raises(IndexError):
        ActionHistory(3).at(0)


def test_truncate_makes_the_cursor_frame_the_newest():
    history = _filled(6)
    history.truncate(2)
    assert len(history) == 4
    assert history.at(0) == {"j": 3.0}
    history.truncate(0)  # no-op
    assert len(history) == 4
    with pytest.raises(IndexError):
        history.truncate(4)


def test_frames_are_copies():
    history = ActionHistory(3)
    action = {"j": 1.0}
    history.push(action)
    action["j"] = 2.0
    out = history.at(0)
    out["j"] = 3.0
    assert history.at(0) == {"j": 1.0}


def test_clear_and_bad_size():
    history = _filled(3)
    history.clear()
    assert len(history) == 0
    with pytest.raises(ValueError):
        ActionHistory(0)
