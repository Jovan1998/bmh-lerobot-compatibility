"""BMH: leader Pi → follower Pi teleoperation over ZMQ.

`bmh/scripts/so_leader_host.py` reads the SO leader arms on the leader Pi and PUSHes their
positions; `SONetworkLeader` on the follower Pi PULLs them as a drop-in `Teleoperator`.
"""

import json
import socket
import sys
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("zmq")

from bmh.scripts import so_leader_host  # noqa: E402
from lerobot.teleoperators.so_leader import so_leader as so_leader_module  # noqa: E402
from lerobot.teleoperators.so_network_leader import (  # noqa: E402
    SONetworkLeader,
    SONetworkLeaderConfig,
)
from lerobot.utils.errors import DeviceNotConnectedError  # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeLeader:
    """Stands in for a connected SOLeader; raises KeyboardInterrupt (host shutdown) on `stop`."""

    def __init__(self, positions: dict[str, float]):
        self.positions = positions
        self.stop = threading.Event()
        self.disconnected = False

    def get_action(self) -> dict[str, float]:
        if self.stop.is_set():
            raise KeyboardInterrupt
        return dict(self.positions)

    def disconnect(self):
        self.disconnected = True


@contextmanager
def _running_host(leader: _FakeLeader, port: int):
    argv = ["so_leader_host", "--left_port", "/dev/fake", "--left_zmq_port", str(port), "--fps", "200"]
    with (
        patch.object(sys, "argv", argv),
        patch.object(so_leader_host, "create_leader", return_value=leader) as create_leader,
    ):
        thread = threading.Thread(target=so_leader_host.main, daemon=True)
        thread.start()
        try:
            yield create_leader
        finally:
            leader.stop.set()
            thread.join(timeout=5)
            assert not thread.is_alive(), "leader host did not shut down"


def test_positions_stream_from_host_to_network_leader():
    port = _free_port()
    positions = {f"{m}.pos": float(i) for i, m in enumerate(ARM + ["head_pan", "head_tilt"])}
    leader = _FakeLeader(positions)

    with _running_host(leader, port) as create_leader:
        net = SONetworkLeader(
            SONetworkLeaderConfig(remote_ip="127.0.0.1", port_zmq=port, connect_timeout_s=5, with_head=True)
        )
        net.connect()
        try:
            assert net.get_action() == positions
            assert set(net.action_features) == set(positions)

            leader.positions = {**positions, "gripper.pos": 42.0}
            deadline = time.monotonic() + 2
            while net.get_action()["gripper.pos"] != 42.0:
                assert time.monotonic() < deadline, "new positions never arrived"
        finally:
            net.disconnect()

    # The host builds the left arm with the head servos by default (BMH-101 wiring).
    create_leader.assert_called_once_with("/dev/fake", "so_leader_left", with_head=True)
    assert leader.disconnected


def test_connect_timeout_is_honoured_when_no_host_answers():
    net = SONetworkLeader(
        SONetworkLeaderConfig(remote_ip="127.0.0.1", port_zmq=_free_port(), connect_timeout_s=1)
    )
    t = time.monotonic()
    with pytest.raises(DeviceNotConnectedError, match="Timeout waiting for leader host"):
        net.connect()
    assert 0.9 <= time.monotonic() - t < 3
    assert not net.is_connected


def test_network_leader_returns_last_positions_when_host_goes_quiet():
    port = _free_port()
    positions = {f"{m}.pos": 1.0 for m in ARM}
    leader = _FakeLeader(positions)
    with _running_host(leader, port):
        net = SONetworkLeader(
            SONetworkLeaderConfig(remote_ip="127.0.0.1", port_zmq=port, connect_timeout_s=5)
        )
        net.connect()
    # Host stopped: the arm holds the last commanded position instead of erroring.
    try:
        assert net.get_action() == positions
        assert net.get_action() == positions
    finally:
        net.disconnect()


# --- create_leader: headless connect on the real SOLeader --------------------------------


def _calibration(motors) -> dict:
    return {
        m: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, m in enumerate(motors, 1)
    }


@contextmanager
def _leader_bus(is_calibrated: bool):
    bus = MagicMock(name="BusMock")
    bus.is_connected = False
    bus.is_calibrated = is_calibrated
    bus.connect.side_effect = lambda: setattr(bus, "is_connected", True)

    def _factory(*_args, **kwargs):
        bus.motors = kwargs["motors"]
        return bus

    bus_cls = "SerialMotorsBus" if hasattr(so_leader_module, "SerialMotorsBus") else "FeetechMotorsBus"
    with (
        patch.object(so_leader_module, bus_cls, side_effect=_factory),
        patch.object(so_leader_module.SOLeader, "configure", lambda self: None),
        patch.object(so_leader_module.SOLeader, "calibrate") as calibrate,
    ):
        yield bus, calibrate


@pytest.fixture
def calibration_root(tmp_path, monkeypatch):
    """`create_leader` uses the default calibration dir; point it at tmp_path."""
    from lerobot.teleoperators.so_leader import config_so_leader

    real_init = config_so_leader.SOLeaderTeleopConfig.__init__

    def _init(self, *args, **kwargs):
        kwargs.setdefault("calibration_dir", tmp_path)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(config_so_leader.SOLeaderTeleopConfig, "__init__", _init)
    return tmp_path


def test_create_leader_requires_a_calibration_file(calibration_root):
    with _leader_bus(is_calibrated=False), pytest.raises(RuntimeError, match="Calibrate this arm first"):
        so_leader_host.create_leader("/dev/fake", "left")


def test_create_leader_rewrites_drifted_registers_without_prompting(calibration_root):
    (calibration_root / "left.json").write_text(json.dumps(_calibration(ARM + ["head_pan", "head_tilt"])))
    with _leader_bus(is_calibrated=False) as (bus, calibrate):
        leader = so_leader_host.create_leader("/dev/fake", "left", with_head=True)

    assert list(leader.bus.motors) == ARM + ["head_pan", "head_tilt"]
    bus.write_calibration.assert_called_once_with(leader.calibration)
    calibrate.assert_not_called()
