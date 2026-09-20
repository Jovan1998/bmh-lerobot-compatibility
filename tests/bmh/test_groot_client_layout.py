"""Unit tests for the BMH-101 GR00T client joint-group layout.

Pure Python + numpy — no robot hardware or ZMQ. Run with
``uv run pytest tests/bmh -svv``.
"""

import numpy as np
import pytest

from bmh.groot_client.layout import (
    FEEDBACK_PREFIX,
    JointGroup,
    group_joint_names,
    group_key_for,
    pack_state,
    split_feedback,
    unpack_action,
    unpack_feedback,
)

# `bi_so_follower.action_features` keys for the 7-DoF + head BMH-101
# (`--robot.left_arm_config.with_head=true`) — identical to the recorded
# dataset's `features["observation.state"].names`.
BMH_101_16_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_yaw.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "left_head_pan.pos",
    "left_head_tilt.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_yaw.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]

# The pre-head 6-DoF layout the client used to hardcode.
LEGACY_12_NAMES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]


def test_group_key_for_classifies_arm_gripper_and_head():
    assert group_key_for("left_shoulder_pan.pos") == "left_single_arm"
    assert group_key_for("left_wrist_yaw.pos") == "left_single_arm"
    assert group_key_for("left_gripper.pos") == "left_gripper"
    assert group_key_for("left_head_pan.pos") == "left_head"
    assert group_key_for("left_head_tilt.pos") == "left_head"
    assert group_key_for("right_gripper.pos") == "right_gripper"


@pytest.mark.parametrize("bad", ["left_shoulder_pan", "head_pan.pos", "left.pos", "center_gripper.pos"])
def test_group_key_for_rejects_malformed_names(bad: str):
    with pytest.raises(ValueError):
        group_key_for(bad)


def test_group_joint_names_16_dim_bmh101():
    groups = group_joint_names(BMH_101_16_NAMES)
    assert groups == [
        JointGroup("left_single_arm", BMH_101_16_NAMES[0:6]),
        JointGroup("left_gripper", ["left_gripper.pos"]),
        JointGroup("left_head", ["left_head_pan.pos", "left_head_tilt.pos"]),
        JointGroup("right_single_arm", BMH_101_16_NAMES[9:15]),
        JointGroup("right_gripper", ["right_gripper.pos"]),
    ]
    # Same keys, same order, same slices as the backend's buildGrootModality.
    assert [g.key for g in groups] == [
        "left_single_arm",
        "left_gripper",
        "left_head",
        "right_single_arm",
        "right_gripper",
    ]
    assert [len(g.names) for g in groups] == [6, 1, 2, 6, 1]


def test_group_joint_names_legacy_12_dim():
    groups = group_joint_names(LEGACY_12_NAMES)
    assert [g.key for g in groups] == [
        "left_single_arm",
        "left_gripper",
        "right_single_arm",
        "right_gripper",
    ]
    assert [len(g.names) for g in groups] == [5, 1, 5, 1]


def test_group_joint_names_rejects_non_contiguous_group():
    names = ["left_shoulder_pan.pos", "left_gripper.pos", "right_shoulder_pan.pos", "left_gripper.pos"]
    with pytest.raises(ValueError, match="left_gripper.*not contiguous"):
        group_joint_names(names)


def test_group_joint_names_rejects_empty():
    with pytest.raises(ValueError):
        group_joint_names([])


def test_pack_state_and_unpack_action_round_trip():
    groups = group_joint_names(BMH_101_16_NAMES)
    obs = {name: float(i) * 1.5 - 4.0 for i, name in enumerate(BMH_101_16_NAMES)}
    obs["left_front"] = np.zeros((360, 640, 3), dtype=np.uint8)  # cameras are ignored
    obs["lang"] = "pick up the cube"

    state = pack_state(obs, groups)
    assert set(state) == {g.key for g in groups}
    assert state["left_single_arm"].dtype == np.float32
    assert state["left_single_arm"].shape == (6,)
    assert state["left_gripper"].shape == (1,)
    assert state["left_head"].shape == (2,)
    assert state["right_single_arm"].shape == (6,)
    assert state["right_gripper"].shape == (1,)
    np.testing.assert_allclose(state["left_head"], [obs["left_head_pan.pos"], obs["left_head_tilt.pos"]])

    # Fake action chunk of shape (B=1, T=24, dim) per group whose timestep t
    # holds the packed state shifted by t, so we can check every t.
    horizon = 24
    chunk = {key: np.stack([np.stack([vec + t for t in range(horizon)])]) for key, vec in state.items()}
    assert chunk["left_single_arm"].shape == (1, horizon, 6)

    for t in (0, 7, horizon - 1):
        action = unpack_action(chunk, groups, t)
        assert list(action) == BMH_101_16_NAMES  # every joint, column order, head included
        for name in BMH_101_16_NAMES:
            assert action[name] == pytest.approx(obs[name] + t, abs=1e-5)
        assert all(isinstance(v, float) for v in action.values())


def test_unpack_action_rejects_wrong_width():
    groups = group_joint_names(BMH_101_16_NAMES)
    chunk = {g.key: np.zeros((1, 4, len(g.names))) for g in groups}
    chunk["left_head"] = np.zeros((1, 4, 1))  # server trained without head_tilt
    with pytest.raises(ValueError, match="left_head"):
        unpack_action(chunk, groups, 0)


# --------------------------------------------------------------------------- feedback


def _chunk_with_feedback(horizon: int = 4) -> dict:
    groups = group_joint_names(BMH_101_16_NAMES)
    chunk = {g.key: np.zeros((1, horizon, len(g.names)), dtype=np.float32) for g in groups}
    # (B=1, T, W=4): the platform duplicates each feedback over W channels.
    done = np.zeros((1, horizon, 4), dtype=np.float32)
    done[0, 1] = [0.0, 0.0, 1.0, 1.0]
    done[0, 2] = [1.0, 1.0, 1.0, 1.0]
    chunk[f"{FEEDBACK_PREFIX}done"] = done
    chunk[f"{FEEDBACK_PREFIX}object_grasped"] = np.full((1, horizon, 2), 0.25, dtype=np.float32)
    return chunk


def test_unpack_action_ignores_feedback_keys():
    # What an old client does with a feedback-trained skill, too: the extra keys are inert.
    action = unpack_action(_chunk_with_feedback(), group_joint_names(BMH_101_16_NAMES), 1)
    assert list(action) == BMH_101_16_NAMES


def test_unpack_feedback_averages_the_channel_width():
    chunk = _chunk_with_feedback()
    assert unpack_feedback(chunk, 0) == {"feedback_done": 0.0, "feedback_object_grasped": 0.25}
    assert unpack_feedback(chunk, 1) == {"feedback_done": 0.5, "feedback_object_grasped": 0.25}
    assert unpack_feedback(chunk, 2) == {"feedback_done": 1.0, "feedback_object_grasped": 0.25}
    assert all(type(v) is float for v in unpack_feedback(chunk, 1).values())


def test_unpack_feedback_is_empty_for_a_skill_without_feedbacks():
    groups = group_joint_names(BMH_101_16_NAMES)
    assert unpack_feedback({g.key: np.zeros((1, 4, len(g.names))) for g in groups}, 0) == {}


def test_split_feedback_separates_joints_from_bare_feedback_keys():
    step = {"left_gripper.pos": 3.0, "feedback_done": 0.9, "right_gripper.pos": -1.0, "feedback_a_b": 0.1}
    action, feedback = split_feedback(step)
    assert action == {"left_gripper.pos": 3.0, "right_gripper.pos": -1.0}
    assert feedback == {"done": 0.9, "a_b": 0.1}  # prefix stripped: the platform's feedback key
    assert split_feedback({"left_gripper.pos": 3.0}) == ({"left_gripper.pos": 3.0}, {})
