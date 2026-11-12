"""`bmh-center-arm --torque_limit`: no motor is ever enabled before its limit is written.

The bench test for the DAgger leader assist drives a leader arm, whose supply is small.
The order of the bus calls is the safety property: every `Torque_Limit` first, then the
motors one at a time, and on release the torque off before the old limits come back.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bmh.scripts import bmh_center_arm
from lerobot.motors import MotorNormMode

MOTORS = ["shoulder_pan", "elbow_flex", "gripper"]


class _Bus:
    def __init__(self, fail_restore_on: str | None = None):
        self.motors = {
            m: SimpleNamespace(
                norm_mode=MotorNormMode.RANGE_0_100 if m == "gripper" else MotorNormMode.DEGREES
            )
            for m in MOTORS
        }
        self.calls: list[tuple] = []
        self.limits = {"shoulder_pan": 1000, "elbow_flex": 1000, "gripper": 500}
        self.is_calibrated = True
        self.fail_restore_on = fail_restore_on
        self.torque_on: set[str] = set()

    def read(self, data_name, motor, *, normalize=True):
        assert data_name == "Torque_Limit" and normalize is False
        self.calls.append(("read", motor))
        return self.limits[motor]

    def write(self, data_name, motor, value, *, normalize=True):
        assert data_name == "Torque_Limit" and normalize is False
        if self.torque_on and value > self.limits[motor]:
            raise AssertionError(f"limit of {motor} raised to {value} while torque is on")
        if self.fail_restore_on == motor and not self.torque_on and ("off",) in self.calls:
            self.calls.append(("write-failed", motor, value))
            raise ConnectionError("no status packet")
        self.calls.append(("write", motor, value))
        self.limits[motor] = value

    def enable_torque(self, motor=None):
        motors = [motor] if motor else list(self.motors)
        self.calls.append(("on", motor))
        self.torque_on.update(motors)

    def disable_torque(self):
        self.calls.append(("off",))
        self.torque_on.clear()

    def sync_read(self, data_name):
        return dict.fromkeys(self.motors, 0.0) | {"gripper": 50.0}

    def sync_write(self, data_name, values):
        self.calls.append(("goal",))


def test_limit_is_written_to_every_motor_before_any_torque():
    bus = _Bus()
    previous = bmh_center_arm.limit_torque(bus, 100)
    with patch.object(bmh_center_arm.time, "sleep") as sleep:
        bmh_center_arm.enable_torque_staggered(bus)
    assert previous == {"shoulder_pan": 1000, "elbow_flex": 1000, "gripper": 500}
    assert bus.calls == [
        *[("read", m) for m in MOTORS],
        *[("write", m, 100) for m in MOTORS],
        *[("on", m) for m in MOTORS],  # one at a time
    ]
    assert [c.args for c in sleep.call_args_list] == [(0.05,)] * 3


def test_restore_writes_each_motors_own_previous_limit_and_survives_a_failure():
    bus = _Bus(fail_restore_on="elbow_flex")
    previous = bmh_center_arm.limit_torque(bus, 100)
    bus.disable_torque()
    bmh_center_arm.restore_torque_limits(bus, previous)
    # The follower gripper's 500 is not raised to 1000; the failing motor does not stop the rest.
    assert bus.limits == {"shoulder_pan": 1000, "elbow_flex": 100, "gripper": 500}


def _run_main(bus: _Bus, *extra: str) -> list[tuple]:
    arm = SimpleNamespace(
        calibration={"x": 1},
        calibration_fpath="calib.json",
        bus=bus,
        connect=lambda calibrate: None,
        disconnect=lambda: None,
    )
    argv = ["bmh-center-arm", "--type", "leader", "--port", "/dev/fake", "--id", "so_leader_left", *extra]
    with (
        patch.object(sys, "argv", argv),
        patch.object(bmh_center_arm, "create_arm", return_value=arm),
        patch.object(bmh_center_arm.time, "sleep"),
        patch("builtins.input", return_value=""),
        pytest.raises(SystemExit) as exit_info,
    ):
        bmh_center_arm.main()
    assert exit_info.value.code == 0
    return bus.calls


def test_main_with_torque_limit_limits_first_and_restores_after_the_release(capsys):
    calls = _run_main(_Bus(), "--torque_limit", "100", "--duration", "0.1")
    kinds = [c[0] for c in calls]
    first_on = kinds.index("on")
    assert kinds[:first_on].count("write") == 3  # every limit is in place before the first torque
    assert ("on", None) not in calls  # never all motors at once
    off = kinds.index("off")
    assert kinds[off + 1 :] == ["write"] * 3  # the old limits come back only after the release
    assert calls[-3:] == [
        ("write", "shoulder_pan", 1000),
        ("write", "elbow_flex", 1000),
        ("write", "gripper", 500),
    ]
    out = capsys.readouterr().out
    assert "TORQUE_LIMIT value=100 motors=3" in out
    assert "CENTERING_DONE ok=true" in out


def test_main_without_the_flag_is_unchanged():
    calls = _run_main(_Bus(), "--duration", "0.1")
    assert ("on", None) in calls  # full torque, all motors, as before
    assert not any(c[0] in ("read", "write") for c in calls)


@pytest.mark.parametrize("value", ["-1", "1001"])
def test_main_rejects_a_limit_outside_the_register_range(value):
    argv = ["bmh-center-arm", "--type", "leader", "--port", "p", "--id", "i", "--torque_limit", value]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exit_info:
        bmh_center_arm.main()
    assert exit_info.value.code == 2
