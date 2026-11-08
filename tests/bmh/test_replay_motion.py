"""`bmh-replay-motion` against fake buses: torque-on without a jump, the phase lines, the
hold, SIGUSR1 = hold where it is, the release (torque off) — and errors as one line."""

import io
import os
import signal
import threading

import pytest

from bmh.motion.file import MotionFile, save_motion
from bmh.motion.player import plan_replay
from bmh.scripts import bmh_replay_motion as replay


class _FakeBus:
    def __init__(self, name: str, log: list[tuple], present: dict[str, float]):
        self.name, self.log, self.present = name, log, present
        self.is_connected = True
        self.goals: list[dict[str, float]] = []

    def sync_read(self, data_name, num_retry=0):
        self.log.append((self.name, f"read {data_name}"))
        return dict(self.present)

    def sync_write(self, data_name, values, num_retry=0):
        if not self.goals:
            self.log.append((self.name, f"write {data_name}"))
        self.goals.append(dict(values))

    def enable_torque(self, num_retry=0):
        self.log.append((self.name, "torque on"))

    def disconnect(self, disable_torque=True):
        self.log.append((self.name, f"disconnect torque_off={disable_torque}"))
        self.is_connected = False


class _FakeArm:
    def __init__(self, arm_id: str, bus: _FakeBus):
        self.id, self.bus = arm_id, bus


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Fake arms patched into the script; a 13-frame, 60 fps recording on two joints."""
    log: list[tuple] = []
    buses = {
        "left": _FakeBus("left", log, {"shoulder_pan": 11.0, "head_pan": 1.0}),
        "right": _FakeBus("right", log, {"shoulder_pan": 22.0}),
    }

    def create_follower(port, arm_id, with_head):
        return _FakeArm(arm_id, buses["left" if arm_id.endswith("_left") else "right"])

    monkeypatch.setattr(replay, "create_follower", create_follower)
    monkeypatch.setattr(replay, "connect_bus", lambda arm: log.append((arm.id, "connect")))
    path = tmp_path / "motion.json"
    save_motion(
        path,
        MotionFile(
            fps=60,
            keys=("left_shoulder_pan.pos", "right_shoulder_pan.pos"),
            frames=tuple((30.0 + i, -float(i)) for i in range(13)),
            recorded_at=0.0,
        ),
    )
    return log, buses, str(path)


def _args(path: str, *extra: str) -> list[str]:
    return ["--left_port", "L", "--right_port", "R", "--file", path, "--fps", "500", *extra]


def test_split_pose():
    assert replay.split_pose({"left_a.pos": 1, "right_b.pos": 2.5, "left_c.vel": 3, "d.pos": 4}) == {
        "left_": {"a": 1.0},
        "right_": {"b": 2.5},
    }


def test_replay_pins_goals_before_torque_plays_holds_and_releases(rig, monkeypatch, capsys):
    log, buses, path = rig
    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    replay.main(_args(path, "--mode", "goto-start", "--ramp_s", "0.05"))

    out = capsys.readouterr().out.splitlines()
    assert out == ["REPLAY_PHASE ramping", "REPLAY_PHASE playing", "REPLAY_DONE"]
    # Each bus: read where it is, pin the goal there, then torque on — before any motion.
    assert log[:8] == [
        ("so_follower_left", "connect"),
        ("so_follower_right", "connect"),
        ("left", "read Present_Position"),
        ("left", "write Goal_Position"),
        ("left", "torque on"),
        ("right", "read Present_Position"),
        ("right", "write Goal_Position"),
        ("right", "torque on"),
    ]
    assert buses["left"].goals[0] == {"shoulder_pan": 11.0, "head_pan": 1.0}  # the pin
    assert buses["left"].goals[1] == {"shoulder_pan": 11.0}  # ramp starts where the arm is
    assert buses["left"].goals[-1] == {"shoulder_pan": 42.0}
    assert buses["right"].goals[-1] == {"shoulder_pan": -12.0}
    assert log[-2:] == [("left", "disconnect torque_off=True"), ("right", "disconnect torque_off=True")]


def test_blend_in_starts_where_the_arms_are(rig, monkeypatch, capsys):
    log, buses, path = rig
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # EOF releases too
    replay.main(_args(path, "--mode", "blend-in", "--blend_s", "0.05", "--start", "2", "--end", "10"))
    assert capsys.readouterr().out.splitlines() == ["REPLAY_PHASE playing", "REPLAY_DONE"]
    assert buses["left"].goals[1] == {"shoulder_pan": 11.0}
    assert buses["left"].goals[-1] == {"shoulder_pan": 39.0}  # frame 9: the range end is exclusive


def test_sigusr1_holds_where_the_arms_are():
    motion = MotionFile(60, ("left_a.pos",), ((0.0,), (10.0,)), 0.0)
    plan = plan_replay({"left_a.pos": -10.0}, motion, 0, 2, "goto-start", ramp_s=1.0)
    bus = _FakeBus("left", [], {})
    hold_now = threading.Event()
    threading.Timer(0.2, hold_now.set).start()
    assert replay.play(plan, {"left_": bus, "right_": bus}, 200, hold_now) is False
    assert -10.0 < bus.goals[-1]["a"] < 0.0  # stopped mid-ramp, not at the end


def test_sigusr1_handler_is_installed(rig, monkeypatch, capsys):
    log, buses, path = rig

    class _Stdin:
        def readline(self):
            return "\n"

    started = threading.Event()
    original_play = replay.play

    def play(plan, buses_, fps, hold_now):
        started.set()
        os.kill(os.getpid(), signal.SIGUSR1)
        return original_play(plan, buses_, fps, hold_now)

    monkeypatch.setattr(replay, "play", play)
    monkeypatch.setattr("sys.stdin", _Stdin())
    replay.main(_args(path, "--ramp_s", "60"))
    assert started.is_set()
    assert capsys.readouterr().out.splitlines()[-1] == "REPLAY_PHASE holding"


@pytest.mark.parametrize(
    ("extra", "message"),
    [(("--end", "99"), "range"), (("--fps", "0"), "fps")],
)
def test_errors_are_one_line_and_release_the_arms(rig, monkeypatch, capsys, extra, message):
    log, buses, path = rig
    with pytest.raises(SystemExit) as exit_info:
        replay.main(_args(path, *extra))
    assert exit_info.value.code == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("ERROR ") and message in out[0]
    assert not buses["left"].is_connected and not buses["right"].is_connected


def test_a_missing_file_is_an_error(rig, capsys):
    log, buses, path = rig
    with pytest.raises(SystemExit):
        replay.main(_args(path + ".gone"))
    assert capsys.readouterr().out.startswith("ERROR ")
