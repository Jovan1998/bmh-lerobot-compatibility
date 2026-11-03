"""The policy client reads feedback channels of any width.

GR00T checkpoints carry 4 duplicated dims per feedback, SmolVLA checkpoints 2; the
client averages whatever arrives, so the same binary drives both families.
"""

import importlib
import sys
import warnings

import numpy as np
import pytest

from bmh.policy_client.layout import split_feedback, unpack_feedback


@pytest.mark.parametrize("width", [2, 4])
def test_unpack_feedback_averages_any_width(width: int) -> None:
    horizon = 3
    done = np.zeros((1, horizon, width), dtype=np.float32)
    done[0, 1, :] = 1.0  # fully on at t=1
    done[0, 2, : width // 2] = 1.0  # half the dims on at t=2
    chunk = {
        "left_single_arm": np.zeros((1, horizon, 6), dtype=np.float32),
        "feedback_done": done,
    }

    assert unpack_feedback(chunk, 0) == {"feedback_done": 0.0}
    assert unpack_feedback(chunk, 1) == {"feedback_done": 1.0}
    assert unpack_feedback(chunk, 2) == {"feedback_done": pytest.approx(0.5)}


def test_split_feedback_keeps_any_width_away_from_the_robot() -> None:
    step = {"left_gripper.pos": 1.0, "feedback_done": 0.5, "feedback_grasped": 1.0}
    joints, feedback = split_feedback(step)
    assert joints == {"left_gripper.pos": 1.0}
    assert feedback == {"done": 0.5, "grasped": 1.0}


def test_groot_client_alias_warns_and_shares_modules() -> None:
    for name in [m for m in sys.modules if m.startswith("bmh.groot_client")]:
        del sys.modules[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("bmh.groot_client")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)

    from bmh.groot_client.feedback import FeedbackTracker as OldTracker

    from bmh.groot_client import layout as old_layout
    from bmh.policy_client import layout
    from bmh.policy_client.feedback import FeedbackTracker

    assert old_layout is layout
    assert OldTracker is FeedbackTracker
