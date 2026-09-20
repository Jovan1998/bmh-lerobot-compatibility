"""Adapter-level tests for ``bmh-groot-client`` — no robot hardware, no ZMQ traffic.

Exercises ``BiSoBimanualAdapter`` end to end on fake observations / action chunks
for the 16-dim 7-DoF + head BMH-101 layout. Needs the ``bmh_network`` extra
(``msgpack-numpy``) because importing the script pulls in ``PolicyClient``; the
module is skipped otherwise. Run with ``uv run pytest tests/bmh -svv``.
"""

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")

from bmh.scripts.bmh_groot_client import BIMANUAL_CAMERA_KEYS, BiSoBimanualAdapter  # noqa: E402
from tests.bmh.test_groot_client_layout import BMH_101_16_NAMES, LEGACY_12_NAMES  # noqa: E402

BMH_101_STATE_KEYS = ("left_single_arm", "left_gripper", "left_head", "right_single_arm", "right_gripper")
LEGACY_STATE_KEYS = ("left_single_arm", "left_gripper", "right_single_arm", "right_gripper")


class _Section:
    """Stand-in for the server's ``ModalityConfig`` (only ``modality_keys`` is read)."""

    def __init__(self, keys):
        self.modality_keys = list(keys)


def _modality_cfg(state_keys, video_keys=tuple(BIMANUAL_CAMERA_KEYS)) -> dict:
    return {"state": _Section(state_keys), "video": _Section(video_keys)}


def _fake_obs() -> dict:
    obs = {name: float(i) - 5.0 for i, name in enumerate(BMH_101_16_NAMES)}
    obs["left_front"] = np.zeros((360, 640, 3), dtype=np.uint8)
    obs["left_left_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["right_right_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    obs["lang"] = "put the cube in the bowl"
    return obs


def _adapter(names=BMH_101_16_NAMES) -> BiSoBimanualAdapter:
    # policy_client is only stored; nothing here talks to a server.
    return BiSoBimanualAdapter(policy_client=None, jpeg_quality=80, joint_names=list(names))


def test_adapter_derives_five_groups_from_robot_joint_names():
    adapter = _adapter()
    assert adapter.state_keys == BMH_101_STATE_KEYS
    assert [len(g.names) for g in adapter.groups] == [6, 1, 2, 6, 1]


def test_obs_to_policy_inputs_packs_state_with_head_and_jpeg_cameras():
    inputs = _adapter().obs_to_policy_inputs(_fake_obs())

    assert set(inputs["state"]) == set(BMH_101_STATE_KEYS)
    assert inputs["state"]["left_single_arm"].shape == (1, 1, 6)  # (B, T, dim)
    assert inputs["state"]["left_head"].shape == (1, 1, 2)
    assert inputs["state"]["right_gripper"].shape == (1, 1, 1)
    np.testing.assert_allclose(inputs["state"]["left_head"][0, 0], [2.0, 3.0])  # columns 7, 8

    assert set(inputs["video"]) == set(BIMANUAL_CAMERA_KEYS)
    assert all(isinstance(v, bytes) and v[:2] == b"\xff\xd8" for v in inputs["video"].values())
    assert inputs["language"]["annotation.human.task_description"] == [["put the cube in the bowl"]]


def test_decode_action_chunk_emits_all_16_pos_keys_including_head():
    adapter = _adapter()
    rng = np.random.default_rng(0)
    chunk = {g.key: rng.random((1, 8, len(g.names)), dtype=np.float32) for g in adapter.groups}

    action = adapter.decode_action_chunk(chunk, t=3)

    assert list(action) == BMH_101_16_NAMES
    assert action["left_head_pan.pos"] == pytest.approx(float(chunk["left_head"][0, 3, 0]))
    assert action["left_head_tilt.pos"] == pytest.approx(float(chunk["left_head"][0, 3, 1]))
    assert action["right_gripper.pos"] == pytest.approx(float(chunk["right_gripper"][0, 3, 0]))
    assert action["left_wrist_yaw.pos"] == pytest.approx(float(chunk["left_single_arm"][0, 3, 4]))


class _FakePolicy:
    """Stand-in for ``PolicyClient``: answers every request with one fixed chunk."""

    def __init__(self, chunk: dict):
        self.chunk = chunk

    def get_action(self, _model_input: dict) -> tuple[dict, dict]:
        return self.chunk, {}


def test_get_action_takes_horizon_from_a_joint_group_and_carries_feedback_per_step():
    adapter = _adapter()
    # The feedback key comes first and is longer than the joint groups: the horizon must
    # not be read off "whatever key is first".
    done = np.zeros((1, 12, 4), dtype=np.float32)
    done[0, 5:] = 1.0
    done[0, 4] = [0.0, 1.0, 1.0, 0.0]
    chunk = {"feedback_done": done}
    chunk.update({g.key: np.full((1, 8, len(g.names)), 2.0, dtype=np.float32) for g in adapter.groups})
    adapter.policy = _FakePolicy(chunk)

    steps = adapter.get_action(_fake_obs())

    assert len(steps) == 8
    assert all(list(step) == [*BMH_101_16_NAMES, "feedback_done"] for step in steps)
    assert [step["feedback_done"] for step in steps] == [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]
    assert steps[3]["left_head_pan.pos"] == 2.0


def test_get_action_without_feedback_keys_yields_joints_only():
    adapter = _adapter()
    adapter.policy = _FakePolicy({g.key: np.zeros((1, 8, len(g.names))) for g in adapter.groups})
    steps = adapter.get_action(_fake_obs())
    assert len(steps) == 8
    assert all(list(step) == BMH_101_16_NAMES for step in steps)


def test_validate_modality_accepts_matching_layout():
    _adapter().validate_modality(_modality_cfg(BMH_101_STATE_KEYS))


def test_validate_modality_accepts_dict_sections():
    cfg = {
        "state": {"modality_keys": list(BMH_101_STATE_KEYS)},
        "video": {"modality_keys": list(BIMANUAL_CAMERA_KEYS)},
    }
    _adapter().validate_modality(cfg)


def test_validate_modality_rejects_checkpoint_trained_without_head():
    with pytest.raises(SystemExit, match=r"state modality keys.*\n.*left_head"):
        _adapter().validate_modality(_modality_cfg(LEGACY_STATE_KEYS))


def test_validate_modality_rejects_head_checkpoint_on_legacy_robot():
    with pytest.raises(SystemExit, match="state modality keys"):
        _adapter(LEGACY_12_NAMES).validate_modality(_modality_cfg(BMH_101_STATE_KEYS))


def test_validate_modality_rejects_video_key_mismatch():
    with pytest.raises(SystemExit, match="video modality keys"):
        _adapter().validate_modality(_modality_cfg(BMH_101_STATE_KEYS, video_keys=("front", "wrist")))


def test_validate_modality_skips_when_server_sends_nothing():
    _adapter().validate_modality({})
    _adapter().validate_modality({"video": _Section(BIMANUAL_CAMERA_KEYS)})
