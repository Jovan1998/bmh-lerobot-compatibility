"""BMH: `bmh-read-pose` prints both arms' present pose without ever moving a servo."""

import json

import pytest

from bmh.scripts.bmh_read_pose import ReadPoseError, connect_bus, pose_line

LEFT = {
    "shoulder_pan": 1.23456,
    "shoulder_lift": -2.0,
    "elbow_flex": 3.0,
    "wrist_flex": 4.0,
    "wrist_yaw": 5.0,
    "wrist_roll": 6.0,
    "gripper": 7.0,
    "head_pan": -8.0,
    "head_tilt": 9.0,
}
RIGHT = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 11.0,
    "elbow_flex": 12.0,
    "wrist_flex": 13.0,
    "wrist_yaw": 14.0,
    "wrist_roll": 15.0,
    "gripper": 99.5,
}


def test_pose_line_prefixes_bimanual_action_keys():
    line = pose_line(LEFT, RIGHT)
    assert line.startswith("POSE {")
    assert "\n" not in line
    pose = json.loads(line.removeprefix("POSE "))
    assert len(pose) == 16
    assert pose["left_shoulder_pan.pos"] == 1.235
    assert pose["left_head_pan.pos"] == -8.0
    assert pose["left_head_tilt.pos"] == 9.0
    assert pose["right_gripper.pos"] == 99.5
    assert not any(key.startswith("right_head") for key in pose)


class _FakeBus:
    def __init__(self, calibrated: bool, torque: int):
        self.is_calibrated = calibrated
        self._torque = torque
        self.is_connected = False
        self.written = None

    def connect(self):
        self.is_connected = True

    def sync_read(self, data_name, normalize=True):
        assert data_name == "Torque_Enable" and normalize is False
        return {"shoulder_pan": 0, "gripper": self._torque}

    def write_calibration(self, calibration):
        self.written = calibration


class _FakeArm:
    id = "so_follower_left"
    calibration_fpath = "/nowhere/so_follower_left.json"

    def __init__(self, bus, calibration=None):
        self.bus = bus
        self.calibration = {"shoulder_pan": object()} if calibration is None else calibration


def test_connect_bus_keeps_matching_calibration():
    arm = _FakeArm(_FakeBus(calibrated=True, torque=1))
    connect_bus(arm)
    assert arm.bus.is_connected and arm.bus.written is None


def test_connect_bus_rewrites_drifted_calibration_only_while_limp():
    arm = _FakeArm(_FakeBus(calibrated=False, torque=0))
    connect_bus(arm)
    assert arm.bus.written is arm.calibration


def test_connect_bus_refuses_drifted_calibration_under_torque():
    arm = _FakeArm(_FakeBus(calibrated=False, torque=1))
    with pytest.raises(ReadPoseError, match="torque is on"):
        connect_bus(arm)
    assert arm.bus.written is None


def test_connect_bus_requires_a_calibration_file():
    arm = _FakeArm(_FakeBus(calibrated=True, torque=0), calibration={})
    with pytest.raises(ReadPoseError, match="no calibration file"):
        connect_bus(arm)
    assert not arm.bus.is_connected
