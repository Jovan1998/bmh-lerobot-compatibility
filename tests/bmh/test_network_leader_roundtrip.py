"""BMH: leader Pi → follower Pi teleoperation over ZMQ.

`bmh/scripts/so_leader_host.py` reads the SO leader arms on the leader Pi and PUSHes their
positions; `SONetworkLeader` on the follower Pi PULLs them as a drop-in `Teleoperator`.
With feedback ports on both sides, `send_feedback` goes the other way: the DAgger leader
assist, driven by the host's `LeaderAssist` on the arm's bus.
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
def _running_host(leader: _FakeLeader, port: int, *extra: str):
    argv = [
        "so_leader_host",
        "--left_port",
        "/dev/fake",
        "--left_zmq_port",
        str(port),
        "--fps",
        "200",
        *extra,
    ]
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


def test_last_message_age_resets_on_a_message_and_grows_when_the_host_goes_quiet():
    port = _free_port()
    leader = _FakeLeader({f"{m}.pos": 1.0 for m in ARM})
    net = SONetworkLeader(SONetworkLeaderConfig(remote_ip="127.0.0.1", port_zmq=port, connect_timeout_s=5))
    assert net.last_message_age_s == float("inf")  # nothing was ever received
    with _running_host(leader, port):
        net.connect()
        assert net.last_message_age_s < 0.5
        time.sleep(0.05)
        net.get_action()  # the host streams at 200 Hz: a fresh message is waiting
        assert net.last_message_age_s < 0.04
    # Host stopped: drain what was still queued, then the cached pose only gets older.
    try:
        net.get_action()
        quiet_since = time.perf_counter()
        time.sleep(0.15)
        before = net.last_message_age_s
        net.get_action()  # returns the cached pose; no message arrived
        assert net.last_message_age_s >= before >= 0.1
        assert net.last_message_age_s <= time.perf_counter() - quiet_since + 0.1
    finally:
        net.disconnect()


def test_bimanual_leader_reports_the_staler_arm():
    from lerobot.teleoperators.bi_so_network_leader import BiSONetworkLeader, BiSONetworkLeaderConfig

    bi = BiSONetworkLeader(BiSONetworkLeaderConfig())
    assert bi.last_message_age_s == float("inf")
    now = time.perf_counter()
    bi.left_arm._last_message_at = now - 0.01
    assert bi.last_message_age_s == float("inf")  # the right arm never spoke
    bi.right_arm._last_message_at = now - 2.0
    assert 2.0 <= bi.last_message_age_s < 2.5


# --- leader assist: follower → leader feedback ---------------------------------------------


class _AssistBus:
    """The leader arm's bus as the host's `LeaderAssist` uses it; records torque and goals."""

    def __init__(self, motors):
        self.motors = dict.fromkeys(motors)
        self.limits = dict.fromkeys(motors, 1000)
        self.torque_on: set[str] = set()
        self.goal: dict[str, float] = {}

    def read(self, data_name, motor, *, normalize=True):
        return self.limits[motor]

    def write(self, data_name, motor, value, *, normalize=True):
        if data_name == "Torque_Limit":
            self.limits[motor] = value
        else:
            self.goal[motor] = value

    def enable_torque(self, motor):
        self.torque_on.add(motor)

    def disable_torque(self, motors):
        self.torque_on.difference_update(motors)

    def sync_write(self, data_name, values):
        self.goal.update(values)


class _AssistLeader(_FakeLeader):
    def __init__(self, positions):
        super().__init__(positions)
        self.bus = _AssistBus([key.removesuffix(".pos") for key in positions])


def _wait_for(condition, what: str, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.005)


def _feedback_pair(port: int, feedback_port: int, *extra: str):
    positions = {f"{m}.pos": 0.0 for m in ARM + ["head_pan", "head_tilt"]}
    leader = _AssistLeader(positions)
    host = _running_host(leader, port, "--left_feedback_port", str(feedback_port), *extra)
    net = SONetworkLeader(
        SONetworkLeaderConfig(
            remote_ip="127.0.0.1",
            port_zmq=port,
            connect_timeout_s=5,
            with_head=True,
            feedback_port_zmq=feedback_port,
        )
    )
    return leader, host, net


def test_feedback_drives_the_arm_joints_and_gripper_never_the_head():
    leader, host, net = _feedback_pair(_free_port(), _free_port(), "--assist_max_torque_limit", "150")
    bus = leader.bus
    with host:
        net.connect()
        try:
            assert set(net.feedback_features) == {f"{m}.pos" for m in ARM}
            goal = {f"{m}.pos": 5.0 for m in ARM} | {"head_pan.pos": 40.0}

            def all_on() -> bool:
                net.send_feedback(goal, torque_limit=900)
                return bus.torque_on == set(ARM)

            _wait_for(all_on, "the arm joints never came on")
            assert {bus.limits[m] for m in ARM} == {150}  # clamped on the leader
            assert bus.limits["head_pan"] == 1000 and "head_pan" not in bus.goal

            net.release_feedback()
            _wait_for(lambda: not bus.torque_on, "release did not turn the torque off")
            assert {bus.limits[m] for m in ARM} == {1000}
        finally:
            net.disconnect()


def test_leader_lets_go_by_itself_when_the_feedback_stops():
    leader, host, net = _feedback_pair(_free_port(), _free_port())
    bus = leader.bus
    with host:
        net.connect()
        try:
            goal = {"shoulder_pan.pos": 1.0}

            def on() -> bool:
                net.send_feedback(goal)
                return "shoulder_pan" in bus.torque_on

            _wait_for(on, "the assist never engaged")
            quiet_since = time.monotonic()
            _wait_for(lambda: not bus.torque_on, "the watchdog never released", timeout_s=2.0)
            assert time.monotonic() - quiet_since >= 0.25
        finally:
            net.disconnect()


def test_disconnect_releases_the_leader():
    leader, host, net = _feedback_pair(_free_port(), _free_port())
    bus = leader.bus
    with host:
        net.connect()

        def on() -> bool:
            net.send_feedback({"gripper.pos": 30.0})
            return "gripper" in bus.torque_on

        _wait_for(on, "the assist never engaged")
        released_by = time.monotonic() + 0.2  # well inside the 0.3 s watchdog
        net.disconnect()
        _wait_for(lambda: not bus.torque_on, "disconnect did not release")
        assert time.monotonic() < released_by


def test_host_releases_on_shutdown():
    leader, host, net = _feedback_pair(_free_port(), _free_port())
    bus = leader.bus
    with host:
        net.connect()

        def on() -> bool:
            net.send_feedback({"shoulder_lift.pos": 3.0})
            return "shoulder_lift" in bus.torque_on

        _wait_for(on, "the assist never engaged")
    # The host thread ended (KeyboardInterrupt) while the assist was on.
    assert not bus.torque_on
    net.disconnect()


def test_send_feedback_without_a_feedback_port_keeps_upstream_behaviour():
    net = SONetworkLeader(SONetworkLeaderConfig(remote_ip="127.0.0.1", port_zmq=_free_port()))
    assert not net.has_feedback and net.feedback_features == {}
    net._is_connected = True
    with pytest.raises(NotImplementedError):
        net.send_feedback({"shoulder_pan.pos": 1.0})
    net.release_feedback()  # nothing to release: a no-op


class _FeedbackArm:
    def __init__(self, keys):
        self.is_connected = True
        self.has_feedback = True
        self.feedback_features = dict.fromkeys(keys, float)
        self.action = dict.fromkeys(keys, 0.0)
        self.sent: list = []

    def send_feedback(self, feedback, torque_limit):
        self.sent.append(("goal", feedback, torque_limit))

    def release_feedback(self):
        self.sent.append(("release",))

    def get_action(self):
        return dict(self.action)


def _bimanual(tmp_path=None):
    from lerobot.teleoperators.bi_so_network_leader import BiSONetworkLeader, BiSONetworkLeaderConfig
    from lerobot.teleoperators.so_network_leader.config_so_network_leader import SONetworkLeaderBaseConfig

    config = BiSONetworkLeaderConfig(
        left_arm_config=SONetworkLeaderBaseConfig(port_zmq=5555, with_head=True, feedback_port_zmq=5556),
        right_arm_config=SONetworkLeaderBaseConfig(port_zmq=5557, feedback_port_zmq=5558),
        lock_file=str(tmp_path / "locks.json") if tmp_path else None,
    )
    bi = BiSONetworkLeader(config)
    assert bi.left_arm.config.feedback_port_zmq == 5556
    assert bi.right_arm.config.feedback_port_zmq == 5558
    assert bi.has_feedback
    bi.left_arm = _FeedbackArm([f"{m}.pos" for m in ARM + ["head_pan", "head_tilt"]])
    bi.right_arm = _FeedbackArm([f"{m}.pos" for m in ARM])
    return bi


def test_bimanual_feedback_splits_by_arm_and_releases_an_arm_left_out():
    bi = _bimanual()
    bi.send_feedback({"left_shoulder_pan.pos": 1.0, "left_gripper.pos": 20.0}, 120)
    assert bi.left_arm.sent == [("goal", {"shoulder_pan.pos": 1.0, "gripper.pos": 20.0}, 120)]
    assert bi.right_arm.sent == [("release",)]
    bi.release_feedback()
    assert bi.left_arm.sent[-1] == ("release",) and bi.right_arm.sent[-1] == ("release",)


def test_bimanual_feedback_needs_both_channels():
    from lerobot.teleoperators.bi_so_network_leader import BiSONetworkLeader, BiSONetworkLeaderConfig
    from lerobot.teleoperators.so_network_leader.config_so_network_leader import SONetworkLeaderBaseConfig

    bi = BiSONetworkLeader(
        BiSONetworkLeaderConfig(left_arm_config=SONetworkLeaderBaseConfig(feedback_port_zmq=5556))
    )
    assert not bi.has_feedback and bi.feedback_features == {}
    bi.left_arm._is_connected = bi.right_arm._is_connected = True
    with pytest.raises(NotImplementedError):
        bi.send_feedback({"left_shoulder_pan.pos": 1.0})


def test_bimanual_locked_keys_follow_the_lock_file(tmp_path):
    bi = _bimanual(tmp_path)
    assert bi.locked_keys == frozenset()
    (tmp_path / "locks.json").write_text(json.dumps({"left": False, "right": True, "head": True}))
    bi.get_action()  # the lock file is read inside get_action
    assert bi.locked_keys == frozenset(
        [f"right_{m}.pos" for m in ARM] + ["left_head_pan.pos", "left_head_tilt.pos"]
    )
    assert _bimanual().locked_keys == frozenset()  # no lock file: nothing is ever locked


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
