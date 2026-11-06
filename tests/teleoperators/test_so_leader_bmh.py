#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BMH: SO leader arm of the BMH-101 — 7-DOF motor layout, optional head, headless connect."""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig, so_leader as so_leader_module

ARM_MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]


def _bus_class_name() -> str:
    # The bus class the leader instantiates (Feetech-specific before upstream 0.6, generic after).
    return "SerialMotorsBus" if hasattr(so_leader_module, "SerialMotorsBus") else "FeetechMotorsBus"


def _make_bus_mock(is_calibrated: bool) -> MagicMock:
    bus = MagicMock(name="BusMock")
    bus.is_connected = False
    bus.is_calibrated = is_calibrated

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@contextmanager
def _patched_leader(tmp_path, *, is_calibrated: bool = True, calibration: dict | None = None, **cfg_kwargs):
    """An `SO101Leader` on a mocked bus; `calibration` (if given) is written as its calibration file."""
    if calibration is not None:
        (tmp_path / "leader.json").write_text(json.dumps(calibration))
    bus_mock = _make_bus_mock(is_calibrated)

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        bus_mock.sync_read.return_value = {m: i for i, m in enumerate(bus_mock.motors, 1)}
        return bus_mock

    with (
        patch.object(so_leader_module, _bus_class_name(), side_effect=_bus_side_effect),
        patch.object(SO101Leader, "configure", lambda self: None),
        patch.object(SO101Leader, "calibrate") as calibrate_mock,
    ):
        cfg = SO101LeaderConfig(port="/dev/null", id="leader", calibration_dir=tmp_path, **cfg_kwargs)
        leader = SO101Leader(cfg)
        leader.calibrate_mock = calibrate_mock
        yield leader
        if leader.is_connected:
            leader.disconnect()


def _calibration_file(motors: list[str]) -> dict:
    return {
        m: {"id": i, "drive_mode": 0, "homing_offset": 10 * i, "range_min": 1000, "range_max": 3000}
        for i, m in enumerate(motors, 1)
    }


# --- motor layout ------------------------------------------------------------------------


_REAL_CONFIGURE = SO101Leader.configure  # `_patched_leader` replaces it with a no-op


def test_configure_retries_the_torque_off(tmp_path):
    with _patched_leader(tmp_path) as leader:
        _REAL_CONFIGURE(leader)

        leader.bus.disable_torque.assert_called_once_with(num_retry=5)


def test_motor_layout(tmp_path):
    # BMH-101 arm: 7 motors, IDs 7 (shoulder_pan) down to 1 (gripper), no head by default
    with _patched_leader(tmp_path) as leader:
        assert list(leader.bus.motors) == ARM_MOTORS
        assert [m.id for m in leader.bus.motors.values()] == [7, 6, 5, 4, 3, 2, 1]
        assert set(leader.action_features) == {f"{m}.pos" for m in ARM_MOTORS}


def test_motor_layout_with_head(tmp_path):
    with _patched_leader(tmp_path, with_head=True) as leader:
        assert list(leader.bus.motors) == ARM_MOTORS + ["head_pan", "head_tilt"]
        assert [leader.bus.motors[m].id for m in ("head_pan", "head_tilt")] == [8, 9]
        assert "head_pan.pos" in leader.action_features
        assert "head_tilt.pos" in leader.action_features


def test_get_action_reads_all_motors(tmp_path):
    with _patched_leader(tmp_path, with_head=True) as leader:
        leader.connect()
        action = leader.get_action()
        assert set(action) == {f"{m}.pos" for m in ARM_MOTORS + ["head_pan", "head_tilt"]}


# --- connect: calibration handling (must work headless, no stdin) ------------------------


def test_connect_applies_drifted_calibration_file_without_prompting(tmp_path):
    with _patched_leader(tmp_path, is_calibrated=False, calibration=_calibration_file(ARM_MOTORS)) as leader:
        assert set(leader.calibration) == set(ARM_MOTORS)
        leader.connect()

        leader.bus.write_calibration.assert_called_once_with(leader.calibration)
        leader.calibrate_mock.assert_not_called()
        assert leader.is_connected


def test_connect_without_calibration_file_runs_calibrate(tmp_path):
    with _patched_leader(tmp_path, is_calibrated=False) as leader:
        assert leader.calibration == {}
        leader.connect()

        leader.calibrate_mock.assert_called_once_with()
        leader.bus.write_calibration.assert_not_called()


def test_connect_when_calibrated_touches_nothing(tmp_path):
    with _patched_leader(tmp_path, is_calibrated=True, calibration=_calibration_file(ARM_MOTORS)) as leader:
        leader.connect()

        leader.calibrate_mock.assert_not_called()
        leader.bus.write_calibration.assert_not_called()


@pytest.mark.parametrize("has_file", [True, False])
def test_connect_calibrate_false_skips_calibration(tmp_path, has_file):
    calibration = _calibration_file(ARM_MOTORS) if has_file else None
    with _patched_leader(tmp_path, is_calibrated=False, calibration=calibration) as leader:
        leader.connect(calibrate=False)

        leader.calibrate_mock.assert_not_called()
        leader.bus.write_calibration.assert_not_called()
