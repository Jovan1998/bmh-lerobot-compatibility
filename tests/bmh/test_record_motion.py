"""`bmh-record-motion`: config decoding, frame extraction, and a whole take against fakes.

The config's `robot` / `teleop` sections are what the controller-app writes for
`bmh-record` (without cameras); they must decode to the same registry types. `run()` is
driven with a fake follower and leader (patched into the lerobot factories): it prints
`MOTION_READY`, records the actions *sent*, saves on stop and disconnects the follower.
"""

import json
import signal

import pytest

pytest.importorskip("draccus")

from bmh.motion.file import MOTION_KEYS, load_motion  # noqa: E402
from bmh.scripts import bmh_record_motion as rec  # noqa: E402
from lerobot.robots.bi_so_follower import BiSOFollowerConfig  # noqa: E402
from lerobot.teleoperators.bi_so_network_leader import BiSONetworkLeaderConfig  # noqa: E402


def _config(output: str, **overrides) -> dict:
    return {
        "robot": {
            "type": "bi_so_follower",
            "id": "so_follower",
            "left_arm_config": {"port": "/dev/ttyACM1", "with_head": True},
            "right_arm_config": {"port": "/dev/ttyACM0"},
        },
        "teleop": {
            "type": "bi_so_network_leader",
            "left_arm_config": {"remote_ip": "192.168.1.2", "port_zmq": 5555, "with_head": True},
            "right_arm_config": {"remote_ip": "192.168.1.2", "port_zmq": 5557},
            "lock_file": "/home/pi/.cache/bmh-101/teleop-locks.json",
        },
        "output": output,
        "fps": 60,
        **overrides,
    }


def _write(tmp_path, data) -> str:
    path = tmp_path / "fixed-recording-config.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_config_decodes_the_apps_robot_and_teleop_sections(tmp_path):
    cfg = rec.load_record_motion_config(_write(tmp_path, _config("/x/motion.json")))
    assert isinstance(cfg.robot, BiSOFollowerConfig)
    assert cfg.robot.left_arm_config.with_head is True and cfg.robot.left_arm_config.cameras == {}
    assert isinstance(cfg.teleop, BiSONetworkLeaderConfig)
    assert cfg.teleop.lock_file == "/home/pi/.cache/bmh-101/teleop-locks.json"
    assert cfg.teleop.right_arm_config.port_zmq == 5557
    assert (cfg.output, cfg.fps) == ("/x/motion.json", 60)


def test_config_fps_defaults_to_60(tmp_path):
    data = _config("/x/motion.json")
    del data["fps"]
    assert rec.load_record_motion_config(_write(tmp_path, data)).fps == 60


@pytest.mark.parametrize(
    "data",
    [
        ["not", "an", "object"],
        {k: v for k, v in _config("/x").items() if k != "output"},
        _config("/x", fps=0),
        _config("", fps=60),
        _config("/x", robot={"type": "no_such_robot"}),
    ],
)
def test_bad_configs_are_rejected(tmp_path, data):
    with pytest.raises(Exception):  # noqa: B017 - draccus raises its own error types
        rec.load_record_motion_config(_write(tmp_path, data))


def test_record_tick_takes_the_sent_values_in_key_order_rounded():
    sent = {"b.pos": 2.00049, "a.pos": -1.23456, "extra.pos": 9.0}
    assert rec.record_tick(sent, ("a.pos", "b.pos")) == (-1.235, 2.0)


def test_record_tick_names_a_missing_key():
    with pytest.raises(rec.RecordMotionError, match="b.pos"):
        rec.record_tick({"a.pos": 1.0}, ("a.pos", "b.pos"))


def test_missing_keys():
    assert rec.missing_keys(dict.fromkeys(MOTION_KEYS, float), MOTION_KEYS) == []
    no_head = {k: float for k in MOTION_KEYS if "head" not in k}
    assert rec.missing_keys(no_head, MOTION_KEYS) == ["left_head_pan.pos", "left_head_tilt.pos"]


# --------------------------------------------------------------------------- a whole take


class _FakeFollower:
    action_features = dict.fromkeys(MOTION_KEYS, float)

    def __init__(self, log: list[str]):
        self.log = log
        self.is_connected = False

    def connect(self):
        self.log.append("robot connect")
        self.is_connected = True

    def send_action(self, action):
        # What the follower reports as sent: here, clipped to 50 (like max_relative_target).
        return {key: min(value, 50.0) for key, value in action.items()}

    def disconnect(self):
        self.log.append("robot disconnect")
        self.is_connected = False


class _FakeLeader:
    def __init__(self, log: list[str], stop: rec._StopFlag, ticks: int):
        self.log, self.stop, self.ticks = log, stop, ticks
        self.calls = 0
        self.is_connected = False

    def connect(self):
        self.log.append("teleop connect")
        self.is_connected = True

    def get_action(self):
        self.calls += 1
        if self.calls == self.ticks:
            self.stop.event.set()  # "SIGTERM" during this tick: it is still recorded
        return {key: float(self.calls * 10) for key in MOTION_KEYS}

    def disconnect(self):
        self.log.append("teleop disconnect")
        self.is_connected = False


def _patch_factories(monkeypatch, follower, leader):
    import lerobot.robots
    import lerobot.teleoperators

    monkeypatch.setattr(lerobot.robots, "make_robot_from_config", lambda cfg: follower)
    monkeypatch.setattr(lerobot.teleoperators, "make_teleoperator_from_config", lambda cfg: leader)


def test_a_take_records_the_sent_actions_saves_and_lets_go(tmp_path, monkeypatch, capsys):
    log: list[str] = []
    stop = rec._StopFlag()
    follower, leader = _FakeFollower(log), _FakeLeader(log, stop, ticks=7)
    _patch_factories(monkeypatch, follower, leader)
    output = tmp_path / "rec" / "motion.json"
    cfg = rec.load_record_motion_config(_write(tmp_path, _config(str(output), fps=500)))

    assert rec.run(cfg, stop) == 7

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "MOTION_READY" and out[-1] == "MOTION_SAVED 7"
    assert log == ["teleop connect", "robot connect", "robot disconnect", "teleop disconnect"]
    motion = load_motion(output)
    assert (motion.fps, motion.keys, len(motion.frames)) == (500, MOTION_KEYS, 7)
    assert [frame[0] for frame in motion.frames] == [10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0]  # as sent


def test_a_follower_without_the_head_fails_before_connecting(tmp_path, monkeypatch):
    log: list[str] = []
    stop = rec._StopFlag()
    follower = _FakeFollower(log)
    follower.action_features = {k: float for k in MOTION_KEYS if "head" not in k}
    _patch_factories(monkeypatch, follower, _FakeLeader(log, stop, ticks=1))
    cfg = rec.load_record_motion_config(_write(tmp_path, _config(str(tmp_path / "m.json"))))
    with pytest.raises(rec.RecordMotionError, match="head"):
        rec.run(cfg, stop)
    assert log == []


def test_signals_abort_while_connecting_and_stop_gracefully_while_recording():
    stop = rec._StopFlag()
    with pytest.raises(KeyboardInterrupt):
        stop(signal.SIGTERM, None)
    stop.recording = True
    stop(signal.SIGTERM, None)
    assert stop.event.is_set()


def test_main_reports_errors_as_one_line(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        rec.main(["--config", str(tmp_path / "missing.json")])
    assert exit_info.value.code == 1
    assert capsys.readouterr().out.startswith("ERROR ")
