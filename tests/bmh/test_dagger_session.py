"""`bmh-dagger` session: the phase machine against a fake robot and leader, on a real dataset.

`DaggerSession` is ticked by hand on a virtual clock, with commands going in as the JSON
lines the controller-app would write and a scripted stand-in for the policy runner (the
real one has its own tests), so every assertion is about exactly which action reached the
robot on which tick. The takes are written through the real `TakeRecorder` into a
`LeRobotDataset` in a temp dir and read back from disk.

`run_session` (set-up, the loop, the clean-up) is run end to end for the paths that must
never lose data: stdin EOF, a stop signal, and a hardware error after an episode was saved.

Needs the `dataset` extra like `test_bmh_record.py`; skipped otherwise.
"""

import glob
import json
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")
pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("zmq")
pytestmark = pytest.mark.dataset

import pandas as pd  # noqa: E402

from bmh.dagger.protocol import ACK_PREFIX, STATUS_PREFIX, parse_line  # noqa: E402
from bmh.dagger.recorder import TakeRecorder, open_dataset  # noqa: E402
from bmh.policy_client.control import LoopSettings  # noqa: E402
from bmh.scripts.bmh_dagger import (  # noqa: E402
    DaggerConfig,
    DaggerSession,
    load_dagger_config,
    run_session,
)
from lerobot.configs.dataset import DatasetRecordConfig  # noqa: E402
from lerobot.datasets import VideoEncodingManager  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
LEFT = [f"left_{m}.pos" for m in ARM + ["head_pan", "head_tilt"]]
RIGHT = [f"right_{m}.pos" for m in ARM]
JOINTS = LEFT + RIGHT
START = {key: float(i) for i, key in enumerate(JOINTS)}
TASK = "put the cube in the bowl"


def _pose(value: float) -> dict[str, float]:
    return dict.fromkeys(JOINTS, value)


def _shifted(pose: dict[str, float], delta: float) -> dict[str, float]:
    return {key: value + delta for key, value in pose.items()}


# --------------------------------------------------------------------------- fakes


class _Bus:
    def __init__(self, robot: "_Robot", prefix: str):
        self._robot = robot
        self._prefix = prefix

    def sync_read(self, data_name: str, num_retry: int = 0) -> dict[str, float]:
        assert data_name == "Present_Position"
        return {
            key[len(self._prefix) : -len(".pos")]: value
            for key, value in self._robot.pose.items()
            if key.startswith(self._prefix)
        }


class _Robot:
    """A `bi_so_follower` stand-in without cameras: it is wherever it was last told to be."""

    name = "fake_bi_so_follower"
    robot_type = "bi_so_follower"
    cameras: dict = {}
    action_features = dict.fromkeys(JOINTS, float)
    observation_features = dict.fromkeys(JOINTS, float)

    def __init__(self, fail_on_send: int | None = None):
        self.pose = dict(START)
        self.sent: list[dict[str, float]] = []
        self.observations = 0
        self.is_connected = False
        self.disconnected = 0
        self.fail_on_send = fail_on_send
        self.left_arm = SimpleNamespace(bus=_Bus(self, "left_"))
        self.right_arm = SimpleNamespace(bus=_Bus(self, "right_"))

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnected += 1

    def get_observation(self) -> dict[str, float]:
        self.observations += 1
        return dict(self.pose)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if self.fail_on_send is not None and len(self.sent) >= self.fail_on_send:
            raise ConnectionError("motor 6: Parsing error")
        self.sent.append(dict(action))
        self.pose = dict(action)
        return action


class _Leader:
    def __init__(self, pose: dict[str, float] | None = None):
        self.pose = dict(pose or START)
        self.last_message_age_s = 0.0
        self.is_connected = False
        self.disconnected = 0
        self.reads = 0

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnected += 1

    def get_action(self) -> dict[str, float]:
        self.reads += 1
        return dict(self.pose)


class _FeedbackLeader(_Leader):
    """A leader stream with the follower-to-leader channel (feedback ports on both arms)."""

    has_feedback = True
    # As `BiSONetworkLeader`: the arm joints and grippers, never the head.
    feedback_features = dict.fromkeys([f"left_{m}.pos" for m in ARM] + [f"right_{m}.pos" for m in ARM], float)

    def __init__(self, pose: dict[str, float] | None = None):
        super().__init__(pose)
        self.locked_keys: frozenset[str] = frozenset()
        self.feedback: list[tuple] = []

    def send_feedback(self, feedback: dict[str, float], torque_limit: int) -> None:
        self.feedback.append(("goal", dict(feedback), torque_limit))

    def release_feedback(self) -> None:
        self.feedback.append(("release",))

    def disconnect(self) -> None:
        self.release_feedback()  # as `SONetworkLeader.disconnect`
        super().disconnect()


class _Runner:
    """Scripted policy: while resumed, tick `n` plays every joint at `100 + n`."""

    def __init__(self):
        self.policy_state = "none"
        self.settings = LoopSettings(seq=0, action_horizon=8, refetch_after=3)
        self.active = False
        self.played = 0
        self.targets: list[tuple] = []
        self.resumed_at: list[dict[str, float]] = []
        self.fires = 0
        self.fail_at: int | None = None
        self.stall = False

    def period_s(self) -> float:
        return self.settings.period_s(30)

    def set_target(self, host, port, lang_instruction, api_token):
        self.targets.append((host, port, lang_instruction, api_token))
        self.policy_state = "ready"

    def apply_settings(self, settings: LoopSettings) -> None:
        self.settings = settings

    def resume(self, pose: dict[str, float]) -> None:
        self.resumed_at.append(dict(pose))
        self.active = True

    def suspend(self) -> None:
        self.active = False

    def tick(self, tick_start: float) -> dict[str, float] | None:
        if not self.active or self.stall:
            return None
        if self.fail_at is not None and self.played >= self.fail_at:
            self.policy_state = "error"
            return None
        self.played += 1
        return _pose(100.0 + self.played)

    def fire(self, get_observation) -> bool:
        self.fires += 1
        return False


class _Rig:
    def __init__(
        self,
        tmp_path,
        *,
        control_fps: int = 60,
        episode_time_s: float = 600,
        leader: _Leader | None = None,
        **cfg,
    ):
        self.root = tmp_path / "recording-staging"
        self.cfg = DaggerConfig(
            robot=None,
            teleop=None,
            dataset=DatasetRecordConfig(
                repo_id="local/dagger-test",
                root=str(self.root),
                fps=30,
                single_task=TASK,
                episode_time_s=episode_time_s,
                push_to_hub=False,
                no_stamp=True,
            ),
            control_fps=control_fps,
            **cfg,
        )
        self.robot = _Robot()
        self.leader = leader if leader is not None else _Leader()
        self.runner = _Runner()
        self.dataset = open_dataset(self.cfg.dataset, self.robot)
        self.recorder = TakeRecorder(self.dataset)
        self.lines: list[str] = []
        self.pending: list = []
        self.stop = False
        self.now = 100.0
        self.next_id = 0
        self.session = DaggerSession(
            self.robot,
            self.leader,
            self.recorder,
            self.runner,
            self.cfg,
            read_commands=self._drain,
            emit=self.lines.append,
            stop_requested=lambda: self.stop,
        )
        self.session.start(dict(START), self.now)

    def _drain(self) -> list:
        items, self.pending = self.pending, []
        return items

    def cmd(self, cmd: str, **fields) -> int:
        self.next_id += 1
        self.pending.append(parse_line(json.dumps({"id": self.next_id, "cmd": cmd, **fields})))
        return self.next_id

    def tick(self, n: int = 1) -> bool:
        alive = True
        for _ in range(n):
            alive = self.session.tick(self.now)
            self.now += self.session.period_s()
        return alive

    def do(self, cmd: str, **fields) -> dict:
        """Send one command, tick once, return its ack."""
        command_id = self.cmd(cmd, **fields)
        self.tick()
        return self.ack(command_id)

    def ack(self, command_id: int) -> dict:
        acks = [json.loads(line[len(ACK_PREFIX) :]) for line in self.lines if line.startswith(ACK_PREFIX)]
        return next(a for a in acks if a["id"] == command_id)

    def statuses(self) -> list[dict]:
        return [
            json.loads(line[len(STATUS_PREFIX) :]) for line in self.lines if line.startswith(STATUS_PREFIX)
        ]

    def phases(self) -> list[str]:
        """Phases in the order they were reported, without repeats."""
        out: list[str] = []
        for status in self.statuses():
            if not out or out[-1] != status["phase"]:
                out.append(status["phase"])
        return out

    def sent_values(self, start: int = 0) -> list[float]:
        return [action[JOINTS[0]] - START[JOINTS[0]] for action in self.robot.sent[start:]]

    def policy_run(self, ticks: int) -> None:
        """Set a policy, run it for `ticks` actions, pause."""
        assert self.do("policy", policy_host="h", policy_port=5555, lang_instruction=TASK)["ok"]
        assert self.do("run")["ok"]  # this tick already plays action 1
        self.tick(ticks - 1)
        assert self.do("pause")["ok"]

    def finalize(self) -> LeRobotDatasetMetadata:
        with VideoEncodingManager(self.dataset):
            self.recorder.discard()
        return LeRobotDatasetMetadata("local/dagger-test", root=self.root)

    def frames(self) -> pd.DataFrame:
        files = sorted(glob.glob(str(self.root / "data" / "**" / "*.parquet"), recursive=True))
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@pytest.fixture
def rig(tmp_path) -> _Rig:
    return _Rig(tmp_path)


# --------------------------------------------------------------------------- paused / running


def test_start_reports_ready_and_holds_the_present_pose(rig):
    assert rig.lines[0] == "DAGGER_READY"
    first = rig.statuses()[0]
    assert first == {
        "phase": "paused",
        "policy": "none",
        "history": 1,
        "cursor": 0,
        "episodes": 0,
        "hold": {k: round(v, 2) for k, v in START.items()},
    }
    rig.tick(10)
    assert rig.robot.sent == [START] * 10
    assert len(rig.session.history) == 1  # holding pushes nothing


def test_run_needs_a_policy(rig):
    assert rig.do("run") == {"id": 1, "ok": False, "error": "no policy set"}
    assert rig.session.phase == "paused"


def test_run_plays_the_policy_and_pause_holds_the_last_action(rig):
    rig.policy_run(5)
    assert rig.runner.targets == [("h", 5555, TASK, "")]
    assert rig.runner.resumed_at == [START]
    assert rig.robot.sent[-5:-1] == [_pose(102.0), _pose(103.0), _pose(104.0), _pose(105.0)]
    assert rig.robot.sent[-1] == _pose(105.0)  # the pause tick holds
    assert len(rig.session.history) == 6  # the start pose + 5 actions
    rig.tick(3)
    assert rig.robot.sent[-3:] == [_pose(105.0)] * 3
    assert rig.session.phase == "paused"
    assert rig.phases() == ["paused", "running", "paused"]


def test_running_holds_while_the_first_chunk_is_on_its_way(rig):
    rig.runner.stall = True
    rig.do("policy", policy_host="h", policy_port=1, lang_instruction=TASK)
    rig.do("run")
    rig.tick(4)
    assert rig.session.phase == "running"
    assert set(map(str, rig.robot.sent)) == {str(START)}
    assert len(rig.session.history) == 1


def test_a_policy_error_while_running_pauses(rig):
    rig.do("policy", policy_host="h", policy_port=1, lang_instruction=TASK)
    rig.runner.fail_at = 3
    rig.do("run")
    rig.tick(5)
    assert rig.session.phase == "paused"
    assert not rig.runner.active
    assert rig.robot.sent[-1] == _pose(103.0)
    assert rig.statuses()[-1]["policy"] == "error"


def test_running_ticks_at_the_policy_period_and_follows_the_speed(rig):
    assert rig.session.period_s() == pytest.approx(1 / 30)
    rig.do("policy", policy_host="h", policy_port=1, lang_instruction=TASK)
    rig.do("run")
    rig.runner.apply_settings(LoopSettings(seq=1, action_horizon=8, refetch_after=3, speed_pct=150))
    assert rig.session.period_s() == pytest.approx(1 / 45)
    rig.do("align", task=TASK)
    assert rig.session.period_s() == pytest.approx(1 / 60)


def test_a_target_set_while_paused_is_serviced_every_tick(rig):
    fires = rig.runner.fires
    rig.tick(3)
    assert rig.runner.fires == fires + 3  # the runner decides whether a probe is due


# --------------------------------------------------------------------------- stepping


def test_step_walks_one_frame_per_tick_and_stops_at_the_cursor(rig):
    rig.policy_run(10)  # history: start, 101 .. 110
    sent = len(rig.robot.sent)
    assert rig.do("step", cursor=4)["ok"]
    assert rig.robot.sent[sent] == _pose(109.0)  # the command's tick already moved one frame
    assert rig.session.phase == "stepping"
    rig.tick(3)
    assert rig.robot.sent[sent : sent + 4] == [_pose(109.0), _pose(108.0), _pose(107.0), _pose(106.0)]
    assert rig.session.phase == "paused" and rig.session.cursor == 4
    rig.tick(3)
    assert rig.robot.sent[-3:] == [_pose(106.0)] * 3  # arrived: holds there
    assert len(rig.session.history) == 11  # stepping keeps the frames ahead

    rig.do("step", cursor=2)  # forward again
    rig.tick(1)
    assert rig.robot.sent[-2:] == [_pose(107.0), _pose(108.0)]
    assert rig.session.cursor == 2
    status = rig.statuses()[-1]
    assert (status["phase"], status["history"], status["cursor"]) == ("paused", 11, 2)
    assert status["hold"][JOINTS[0]] == 108.0


def test_a_new_cursor_mid_walk_retargets_without_a_jump(rig):
    rig.policy_run(10)
    rig.do("step", cursor=8)  # -> 109
    rig.tick(2)  # 108, 107
    rig.do("step", cursor=1)  # turns around: 108
    rig.tick(1)  # 109
    values = [a[JOINTS[0]] for a in rig.robot.sent[-5:]]
    assert values == [109.0, 108.0, 107.0, 108.0, 109.0]
    assert rig.session.phase == "paused" and rig.session.cursor == 1


def test_step_past_the_oldest_frame_stops_there(rig):
    rig.policy_run(3)
    assert rig.do("step", cursor=500)["ok"]
    rig.tick(10)
    assert rig.session.cursor == 3
    assert rig.robot.sent[-1] == START


def test_step_to_the_current_cursor_stays_paused(rig):
    rig.policy_run(3)
    assert rig.do("step", cursor=0)["ok"]
    assert rig.session.phase == "paused"


def test_run_after_a_rewind_truncates_the_history(rig):
    rig.policy_run(10)
    rig.do("step", cursor=4)
    rig.tick(5)
    assert rig.do("run")["ok"]
    assert rig.runner.resumed_at[-1] == _pose(106.0)  # the policy restarts from the rewound pose
    assert rig.session.cursor == 0
    # start + 101..106, then the action this tick played.
    assert len(rig.session.history) == 8
    assert rig.session.history.at(1) == _pose(106.0)
    assert rig.session.history.at(0) == _pose(111.0)


def test_pause_mid_walk_stays_where_the_robot_is(rig):
    rig.policy_run(10)
    rig.do("step", cursor=8)
    rig.tick(1)
    assert rig.do("pause")["ok"]
    assert rig.session.phase == "paused" and rig.session.cursor == 2
    assert rig.robot.sent[-1] == _pose(108.0)


# --------------------------------------------------------------------------- home


def test_home_ramps_from_the_hold_pose_and_restarts_the_history(rig):
    rig.policy_run(5)
    home = {JOINTS[0]: 5.0, JOINTS[1]: -20.0, "not_a_joint.pos": 1.0}
    sent = len(rig.robot.sent)
    assert rig.do("home", pose=home)["ok"]
    assert rig.robot.sent[sent] == _pose(105.0)  # the first ramp tick is the start pose
    assert rig.session.phase == "homing"
    rig.tick(80)  # 2.5 s at 30 Hz = 75 ticks
    assert rig.session.phase == "paused"
    first = [a[JOINTS[0]] for a in rig.robot.sent[sent:]]
    assert all(b <= a for a, b in zip(first, first[1:], strict=False))  # monotonic 105 -> 5
    final = rig.robot.sent[-1]
    assert final[JOINTS[0]] == 5.0 and final[JOINTS[1]] == -20.0
    assert final[JOINTS[2]] == 105.0  # joints Home does not list stay
    assert "not_a_joint.pos" not in final
    assert len(rig.session.history) == 1 and rig.session.cursor == 0
    assert rig.session.hold == final


def test_home_with_no_joint_of_this_robot_is_refused(rig):
    ack = rig.do("home", pose={"other.pos": 1.0})
    assert ack["ok"] is False and "names no joint" in ack["error"]


def test_pause_mid_ramp_holds_where_the_ramp_was(rig):
    rig.policy_run(5)
    rig.do("home", pose=_pose(0.0))
    rig.tick(30)
    assert rig.do("pause")["ok"]
    assert rig.session.phase == "paused"
    where = rig.robot.sent[-1]
    assert 0.0 < where[JOINTS[0]] < 105.0
    rig.tick(3)
    assert rig.robot.sent[-3:] == [where] * 3
    assert len(rig.session.history) == 1


# --------------------------------------------------------------------------- aligning


def _align_status(rig) -> dict:
    return next(s for s in reversed(rig.statuses()) if s["phase"] == "aligning")


def test_align_holds_and_reports_leader_minus_hold(rig):
    rig.policy_run(5)
    rig.leader.pose = {**_pose(105.0), JOINTS[0]: 95.0, "left_gripper.pos": 107.5}
    assert rig.do("align", task=TASK, auto_start=False)["ok"]
    rig.tick(30)
    assert rig.session.phase == "aligning"
    assert rig.robot.sent[-30:] == [_pose(105.0)] * 30  # the robot does not follow yet
    status = _align_status(rig)
    assert status["align"]["diff"][JOINTS[0]] == -10.0
    assert status["align"]["diff"]["left_gripper.pos"] == 2.5
    assert status["align"]["aligned"] is False
    assert status["align"]["assist"] == "off"
    assert status["hold"][JOINTS[0]] == 105.0
    assert status["take"] == {"frames": 0}


def test_align_from_running_pauses_the_policy_first(rig):
    rig.do("policy", policy_host="h", policy_port=1, lang_instruction=TASK)
    rig.do("run")
    rig.tick(3)
    rig.leader.pose = _pose(0.0)
    assert rig.do("align", task=TASK)["ok"]
    assert not rig.runner.active
    assert rig.session.phase == "aligning"
    assert rig.robot.sent[-1] == _pose(104.0)


def test_recording_starts_by_itself_only_after_the_gate(rig):
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 10.0)  # far off
    rig.do("align", task=TASK)
    rig.tick(60)
    assert rig.session.phase == "aligning"
    rig.leader.pose = _shifted(_pose(105.0), 3.0)  # inside the 4 degree tolerance
    rig.tick(20)  # 0.33 s at 60 Hz: not yet
    assert rig.session.phase == "aligning"
    rig.leader.pose = _shifted(_pose(105.0), 10.0)  # a gap restarts the 0.4 s
    rig.tick(1)
    rig.leader.pose = _shifted(_pose(105.0), 3.0)
    rig.tick(23)
    assert rig.session.phase == "aligning"
    rig.tick(3)
    assert rig.session.phase == "recording"


def test_auto_start_off_waits_for_record(rig):
    rig.policy_run(5)
    rig.leader.pose = _pose(105.0)
    rig.do("align", task=TASK, auto_start=False)
    rig.tick(60)
    assert rig.session.phase == "aligning"
    assert _align_status(rig)["align"]["aligned"] is True
    assert rig.do("record")["ok"]
    assert rig.session.phase == "recording"


def test_a_stale_leader_never_starts_a_take(rig):
    rig.policy_run(5)
    rig.leader.pose = _pose(105.0)
    rig.leader.last_message_age_s = 0.8
    rig.do("align", task=TASK)
    rig.tick(60)
    assert rig.session.phase == "aligning"
    assert _align_status(rig)["align"]["leader_age_ms"] == 800
    assert rig.do("record") == {"id": rig.next_id, "ok": False, "error": "the leader stream is stale"}
    rig.leader.last_message_age_s = 0.01
    rig.tick(30)
    assert rig.session.phase == "recording"


def test_assist_is_reported_unavailable_without_the_feedback_channel(rig):
    rig.do("align", task=TASK, auto_start=False, assist=True)
    rig.tick(6)
    assert _align_status(rig)["align"]["assist"] == "unavailable"


@pytest.fixture
def assist_rig(tmp_path) -> _Rig:
    return _Rig(tmp_path, leader=_FeedbackLeader())


def _goals(rig) -> list[tuple]:
    return [f for f in rig.leader.feedback if f[0] == "goal"]


ARM_KEYS = [f"left_{m}.pos" for m in ARM] + [f"right_{m}.pos" for m in ARM]


def test_assist_pulls_the_leader_to_the_hold_pose_of_the_arm_joints_only(assist_rig):
    rig = assist_rig
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 30.0)  # far off: no auto-start
    assert rig.do("align", task=TASK, assist=True, assist_torque_pct=15)["ok"]
    rig.tick(9)
    goals = _goals(rig)
    assert len(goals) == 10  # every aligning tick
    assert goals[0] == ("goal", dict.fromkeys(ARM_KEYS, 105.0), 150)  # grippers included, no head
    assert _align_status(rig)["align"]["assist"] == "on"
    assert rig.leader.feedback[-1][0] == "goal"  # not released while aligning


def test_assist_lets_go_the_tick_the_take_starts(assist_rig):
    rig = assist_rig
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 2.0)  # inside tolerance: auto-start after 0.4 s
    rig.do("align", task=TASK, assist=True)
    rig.tick(30)
    assert rig.session.phase == "recording"
    goal_ticks = len(_goals(rig))
    assert rig.leader.feedback[-1] == ("release",)
    assert rig.leader.feedback[goal_ticks:] == [("release",)]  # once, right after the last goal
    rig.tick(30)
    assert len(rig.leader.feedback) == goal_ticks + 1  # nothing more while recording


@pytest.mark.parametrize("how", ["cancel", "record", "finish", "stop"])
def test_every_exit_from_aligning_releases_the_leader(assist_rig, how):
    rig = assist_rig
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 30.0)
    rig.do("align", task=TASK, assist=True, auto_start=False)
    rig.tick(3)
    if how == "stop":
        rig.stop = True
        rig.tick()
    else:
        assert rig.do(how)["ok"]
    assert rig.leader.feedback[-1] == ("release",)


def test_assist_command_switches_it_and_changes_the_torque_while_aligning(assist_rig):
    rig = assist_rig
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 30.0)
    rig.do("align", task=TASK, auto_start=False)  # assist off
    rig.tick(3)
    assert rig.leader.feedback == []
    assert _align_status(rig)["align"]["assist"] == "off"

    assert rig.do("assist", assist=True, assist_torque_pct=20)["ok"]
    assert _goals(rig)[-1][2] == 200
    assert rig.do("assist", assist=True, assist_torque_pct=5)["ok"]
    assert _goals(rig)[-1][2] == 50
    assert rig.do("assist", assist=False)["ok"]
    assert rig.leader.feedback[-1] == ("release",)
    rig.tick(5)
    assert rig.leader.feedback[-1] == ("release",)  # stays off
    rig.tick(6)
    assert _align_status(rig)["align"]["assist"] == "off"


def test_assist_command_outside_aligning_is_refused(assist_rig):
    assert assist_rig.do("assist", assist=True) == {
        "id": assist_rig.next_id,
        "ok": False,
        "error": "not aligning",
    }
    assert assist_rig.leader.feedback == []


def test_a_locked_arm_is_not_pulled(assist_rig):
    rig = assist_rig
    rig.policy_run(5)
    rig.leader.pose = _shifted(_pose(105.0), 30.0)
    rig.leader.locked_keys = frozenset(f"right_{m}.pos" for m in ARM)
    rig.do("align", task=TASK, assist=True, auto_start=False)
    assert set(_goals(rig)[-1][1]) == {f"left_{m}.pos" for m in ARM}


def test_cancel_returns_to_paused_at_the_same_cursor(rig):
    rig.policy_run(10)
    rig.do("step", cursor=3)
    rig.tick(4)
    rig.do("align", task=TASK, auto_start=False)
    assert rig.do("cancel")["ok"]
    assert rig.session.phase == "paused" and rig.session.cursor == 3
    assert len(rig.session.history) == 11  # nothing was truncated


def test_status_goes_out_at_2hz_and_at_10hz_while_aligning(rig):
    rig.tick(60)  # ticks 0 .. 59 at 30 Hz: the start line, then one every 15 ticks
    assert len(rig.statuses()) == 1 + 3
    before = len(rig.statuses())
    rig.do("align", task=TASK, auto_start=False)
    rig.tick(59)  # 1 s at 60 Hz
    aligning = rig.statuses()[before:]
    assert {s["phase"] for s in aligning} == {"aligning"}
    assert 10 <= len(aligning) <= 11


# --------------------------------------------------------------------------- recording


def _start_take(rig, leader: dict[str, float], hold_value: float = 105.0) -> int:
    """Run to `hold_value`, pause, take over with `record`. Returns the index of the first take action."""
    rig.policy_run(int(hold_value - 100))
    rig.leader.pose = dict(leader)
    rig.do("align", task=TASK, auto_start=False)
    rig.tick(3)
    sent = len(rig.robot.sent)
    assert rig.do("record")["ok"]
    return sent


def test_the_first_recorded_action_is_the_hold_pose(rig):
    leader = _shifted(_pose(105.0), 20.0)  # forced from 20 degrees away
    first = _start_take(rig, leader)
    assert rig.robot.sent[first] == _pose(105.0)  # exactly: no jump at the start
    rig.tick(120)  # 2 s; the blend takes max(1, 20 / 45) = 1 s
    values = [a[JOINTS[0]] for a in rig.robot.sent[first:]]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    assert max(steps) < 1.0  # 20 degrees over 60 ticks, smoothstep peak 1.5x the mean
    assert rig.robot.sent[-1] == leader  # after the blend the leader passes through untouched

    assert rig.do("save")["ok"]
    rig.tick(2)
    rig.finalize()
    frames = rig.frames()
    assert list(frames["action"].iloc[0]) == [105.0] * len(JOINTS)
    assert list(frames["action"].iloc[-1]) == [125.0] * len(JOINTS)


def test_the_leader_is_followed_at_control_fps_and_recorded_at_dataset_fps(rig):
    first = _start_take(rig, _pose(105.0))
    observations = rig.robot.observations
    reads = rig.leader.reads
    rig.tick(59)  # 60 take ticks with the record command's
    assert len(rig.robot.sent) - first == 60
    assert rig.leader.reads - reads == 59
    assert rig.recorder.frames == 30
    assert rig.robot.observations - observations == 29  # only record ticks observe
    assert rig.statuses()[-1]["take"] == {"frames": rig.statuses()[-1]["take"]["frames"]}
    assert rig.statuses()[-1]["phase"] == "recording"


def test_save_writes_one_episode_with_the_task_and_returns_to_paused(rig):
    first = _start_take(rig, _pose(105.0))
    rig.tick(29)
    rig.leader.pose = _pose(111.0)
    rig.tick(30)
    assert rig.do("save")["ok"]
    assert rig.session.phase == "saving"
    assert rig.recorder.episodes == 0  # the tick that reports `saving` does not block yet
    rig.tick()
    assert rig.session.phase == "paused"
    assert rig.recorder.episodes == 1
    assert rig.phases()[-3:] == ["recording", "saving", "paused"]
    last = rig.statuses()[-1]
    assert (last["episodes"], last["cursor"]) == (1, 0)
    assert last["hold"][JOINTS[0]] == 111.0  # holds where the take ended
    rig.tick(3)
    assert rig.robot.sent[-3:] == [_pose(111.0)] * 3
    # 60 take ticks, the tick that reported `saving`, 3 paused ticks; the saving tick sends nothing.
    assert len(rig.robot.sent) - first == 60 + 1 + 3

    meta = rig.finalize()
    assert meta.total_episodes == 1
    assert meta.total_frames == 30
    assert list(meta.tasks.index) == [TASK]
    frames = rig.frames()
    assert set(frames["episode_index"]) == {0}
    assert list(frames["observation.state"].iloc[0]) == [105.0] * len(JOINTS)


def test_discard_writes_nothing(rig):
    _start_take(rig, _pose(105.0))
    rig.tick(40)
    assert rig.do("discard")["ok"]
    rig.tick(2)
    assert rig.session.phase == "paused"
    assert rig.recorder.episodes == 0 and rig.recorder.frames == 0
    assert rig.finalize().total_episodes == 0


def test_two_takes_are_two_episodes_and_the_second_starts_from_its_own_hold(rig):
    _start_take(rig, _pose(105.0))
    rig.tick(20)
    rig.do("save")
    rig.tick(2)
    rig.leader.pose = _pose(90.0)
    rig.do("align", task="second task", auto_start=False)
    rig.tick(3)
    first = len(rig.robot.sent)
    rig.do("record")
    assert rig.robot.sent[first] == _pose(105.0)
    rig.tick(100)
    rig.do("save")
    rig.tick(2)
    meta = rig.finalize()
    assert meta.total_episodes == 2
    assert sorted(meta.tasks.index) == sorted([TASK, "second task"])


def test_a_take_from_a_rewound_cursor_truncates_the_history(rig):
    rig.policy_run(10)
    rig.do("step", cursor=6)
    rig.tick(7)
    rig.leader.pose = _pose(104.0)
    rig.do("align", task=TASK, auto_start=False)
    first = len(rig.robot.sent)
    rig.do("record")
    assert rig.robot.sent[first - 1] == _pose(104.0)
    assert rig.session.cursor == 0
    rig.tick(9)
    # start + 101..104, then the take's record ticks (every 2nd of 10).
    assert len(rig.session.history) == 5 + 5


def test_a_take_is_saved_by_itself_at_episode_time_s(tmp_path):
    rig = _Rig(tmp_path, episode_time_s=1)
    _start_take(rig, _pose(105.0))
    rig.tick(70)
    assert rig.session.phase == "paused"
    assert rig.recorder.episodes == 1
    assert rig.finalize().total_frames == 30


def test_save_without_a_frame_writes_nothing(rig):
    rig.do("align", task=TASK, auto_start=False)
    rig.cmd("record")
    rig.cmd("save")
    rig.tick(3)
    assert rig.session.phase == "paused"
    assert rig.recorder.episodes == 0


# --------------------------------------------------------------------------- refusals / end


@pytest.mark.parametrize(
    "setup, cmd, fields, error",
    [
        ([], "save", {}, "not recording"),
        ([], "discard", {}, "not recording"),
        ([], "record", {}, "not aligning"),
        ([], "cancel", {}, "not aligning"),
        ([], "pause", {}, "nothing to pause"),
        (["policy", "run"], "step", {"cursor": 2}, "not paused"),
        (["policy", "run"], "home", {"pose": {JOINTS[0]: 0.0}}, "not paused"),
        (["policy", "run"], "run", {}, "not paused"),
        (["align"], "run", {}, "not paused"),
        (["align"], "align", {"task": "x"}, "not paused or running"),
        (
            ["align"],
            "policy",
            {"policy_host": "h", "policy_port": 1, "lang_instruction": "x"},
            "not during a take-over",
        ),
        (["align", "record"], "step", {"cursor": 1}, "not paused"),
        (["align", "record"], "record", {}, "not aligning"),
    ],
)
def test_a_command_in_the_wrong_phase_is_refused_not_queued(rig, setup, cmd, fields, error):
    presets = {
        "policy": {"policy_host": "h", "policy_port": 1, "lang_instruction": TASK},
        "align": {"task": TASK, "auto_start": False},
    }
    for step in setup:
        assert rig.do(step, **presets.get(step, {}))["ok"]
    phase = rig.session.phase
    ack = rig.do(cmd, **fields)
    assert ack == {"id": rig.next_id, "ok": False, "error": error}
    rig.tick(3)
    assert rig.session.phase == phase


def test_an_invalid_line_is_acked_with_its_id_and_changes_nothing(rig):
    rig.pending.append(parse_line('{"id": 41, "cmd": "dance"}'))
    rig.pending.append(parse_line("garbage"))
    rig.tick()
    assert rig.ack(41) == {"id": 41, "ok": False, "error": "unknown cmd 'dance'"}
    assert rig.session.phase == "paused"


def test_commands_in_one_tick_are_applied_in_order(rig):
    ids = [
        rig.cmd("policy", policy_host="h", policy_port=1, lang_instruction=TASK),
        rig.cmd("run"),
        rig.cmd("pause"),
        rig.cmd("run"),
    ]
    rig.tick()
    assert [rig.ack(i)["ok"] for i in ids] == [True] * 4
    assert rig.session.phase == "running"
    assert len(rig.runner.resumed_at) == 2


def test_finish_right_after_save_still_saves_the_take(rig):
    _start_take(rig, _pose(105.0))
    rig.tick(20)
    rig.cmd("save")
    rig.cmd("finish")  # same tick: the save has not run yet
    assert rig.tick() is False
    assert rig.recorder.episodes == 1
    assert rig.finalize().total_frames == 11


@pytest.mark.parametrize("how", ["finish", "eof", "signal"])
def test_finish_eof_and_a_stop_signal_end_the_loop(rig, how):
    from bmh.dagger.protocol import Eof

    rig.policy_run(3)
    if how == "finish":
        command_id = rig.cmd("finish")
    elif how == "eof":
        rig.pending.append(Eof())
    else:
        rig.stop = True
    assert rig.tick() is False
    assert rig.statuses()[-1]["phase"] == "finishing"
    if how == "finish":
        assert rig.ack(command_id)["ok"]


def test_finish_mid_take_leaves_the_take_to_be_discarded(rig):
    _start_take(rig, _pose(105.0))
    rig.tick(20)
    rig.cmd("finish")
    assert rig.tick() is False
    assert rig.recorder.frames == 11  # still pending: `run_session` discards it
    assert rig.finalize().total_episodes == 0


# --------------------------------------------------------------------------- loop settings


class _LoopWatcher:
    def __init__(self):
        self.next: LoopSettings | None = None

    def poll(self) -> LoopSettings | None:
        out, self.next = self.next, None
        return out


def test_live_loop_settings_are_applied_and_answered(rig, caplog):
    watcher = _LoopWatcher()
    rig.session._loop_watcher = watcher
    watcher.next = LoopSettings(seq=4, action_horizon=24, refetch_after=10, speed_pct=120, smoothing_pct=30)
    with caplog.at_level("INFO"):
        rig.tick()
    assert rig.runner.settings.seq == 4
    assert 'BMH_LOOP {"seq": 4, "accepted": true' in caplog.text

    watcher.next = LoopSettings(seq=5, action_horizon=8, refetch_after=8)  # refetch must be < horizon
    with caplog.at_level("INFO"):
        rig.tick()
    assert rig.runner.settings.seq == 4  # rejected whole
    assert 'BMH_LOOP {"seq": 5, "accepted": false' in caplog.text


# --------------------------------------------------------------------------- config


def test_config_decodes_the_recording_blocks_and_its_own_keys(tmp_path):
    path = tmp_path / "dagger-config.json"
    path.write_text(
        json.dumps(
            {
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
                "dataset": {
                    "repo_id": "local/my-set",
                    "root": "/tmp/staging",
                    "fps": 30,
                    "single_task": "Place the toast",
                    "video_encoding_batch_size": 50,
                    "rgb_encoder": {"crf": 25, "g": 6},
                    "push_to_hub": False,
                },
                "control_fps": 60,
                "loop_file": "/home/pi/.cache/bmh-101/inference-loop.json",
                "jpeg_quality": 80,
            }
        )
    )
    cfg = load_dagger_config(path)
    assert cfg.robot.type == "bi_so_follower"
    assert cfg.teleop.type == "bi_so_network_leader"
    assert cfg.teleop.lock_file.endswith("teleop-locks.json")
    assert cfg.dataset.video_encoding_batch_size == 50
    assert (cfg.control_fps, cfg.control_multiplier, cfg.jpeg_quality) == (60, 2, 80)
    assert cfg.loop_file.endswith("inference-loop.json")
    assert (cfg.history_s, cfg.align_arm_tol_deg, cfg.align_hold_s, cfg.takeover_blend_s) == (60, 4, 0.4, 1.0)


def test_config_rejects_a_control_rate_that_is_no_multiple_of_the_dataset_fps(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "robot": {
                    "type": "bi_so_follower",
                    "left_arm_config": {"port": "a"},
                    "right_arm_config": {"port": "b"},
                },
                "teleop": {"type": "bi_so_network_leader"},
                "dataset": {"repo_id": "local/x", "fps": 30},
                "control_fps": 45,
            }
        )
    )
    with pytest.raises(ValueError, match="control_fps"):
        load_dagger_config(path)


# --------------------------------------------------------------------------- run_session


class _Stream:
    """stdin stand-in: iterating blocks until the test feeds a line; `None` is EOF."""

    def __init__(self):
        self.lines: queue.Queue[str | None] = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.lines.get(timeout=30)
        if line is None:
            raise StopIteration
        return line


class _Script:
    """Virtual clock for `run_session`: each sleep is one tick and may feed stdin a line."""

    def __init__(self, stream: _Stream, steps: dict[int, str | None]):
        self.stream = stream
        self.steps = steps
        self.now = 0.0
        self.ticks = 0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.ticks += 1
        if self.ticks in self.steps:
            self.stream.lines.put(self.steps[self.ticks])
            # Let the reader thread hand the line over before the next tick drains.
            import time

            time.sleep(0.05)
        assert self.ticks < 5000, "the session never ended"


def _session_cfg(tmp_path) -> DaggerConfig:
    return DaggerConfig(
        robot=None,
        teleop=None,
        dataset=DatasetRecordConfig(
            repo_id="local/dagger-test",
            root=str(tmp_path / "recording-staging"),
            fps=30,
            single_task=TASK,
            push_to_hub=False,
            no_stamp=True,
        ),
    )


def _cmd(command_id: int, cmd: str, **fields) -> str:
    return json.dumps({"id": command_id, "cmd": cmd, **fields}) + "\n"


TAKE = {
    5: _cmd(1, "align", task=TASK, auto_start=False),
    10: _cmd(2, "record"),
    80: _cmd(3, "save"),
}


def _run(
    tmp_path,
    robot: _Robot,
    steps: dict[int, str | None],
    stop_after: int | None = None,
    leader: _Leader | None = None,
):
    stream = _Stream()
    script = _Script(stream, steps)
    leader = leader if leader is not None else _Leader()
    lines: list[str] = []
    code = run_session(
        _session_cfg(tmp_path),
        robot,
        leader,
        stream=stream,
        emit=lines.append,
        stop_requested=lambda: stop_after is not None and script.ticks >= stop_after,
        policy_connect=lambda target: SimpleNamespace(close=lambda: None),
        clock=script.clock,
        sleep=script.sleep,
    )
    stream.lines.put(None)  # release the reader thread if it is still waiting
    meta = LeRobotDatasetMetadata("local/dagger-test", root=tmp_path / "recording-staging")
    return code, lines, leader, meta


def test_eof_finalizes_and_disconnects(tmp_path):
    robot = _Robot()
    code, lines, leader, meta = _run(tmp_path, robot, {3: None})
    assert code == 0
    assert lines[0] == "DAGGER_READY"
    assert not any(line.startswith("ERROR") for line in lines)
    assert json.loads(lines[-1][len(STATUS_PREFIX) :])["phase"] == "finishing"
    assert robot.sent and set(map(str, robot.sent)) == {str(START)}  # held the present pose
    assert (robot.disconnected, leader.disconnected) == (1, 1)
    assert meta.total_episodes == 0


def test_a_saved_take_survives_a_stop_signal_mid_take(tmp_path):
    robot = _Robot()
    steps = {**TAKE, 90: _cmd(4, "align", task=TASK, auto_start=False), 95: _cmd(5, "record")}
    code, lines, leader, meta = _run(tmp_path, robot, steps, stop_after=130)
    assert code == 0
    assert meta.total_episodes == 1  # the second take was running: discarded
    assert meta.total_frames > 20
    assert (robot.disconnected, leader.disconnected) == (1, 1)


def test_a_hardware_error_still_finalizes_and_exits_1(tmp_path):
    # The bus dies mid-way through a second take; the first one is already saved.
    robot = _Robot(fail_on_send=140)
    steps = {**TAKE, 90: _cmd(4, "align", task=TASK, auto_start=False), 95: _cmd(5, "record")}
    code, lines, leader, meta = _run(tmp_path, robot, steps)
    assert code == 1
    assert lines[-1] == "ERROR ConnectionError: motor 6: Parsing error"
    assert meta.total_episodes == 1
    assert list(meta.tasks.index) == [TASK]
    assert (robot.disconnected, leader.disconnected) == (1, 1)


def test_a_hardware_error_while_the_assist_pulls_releases_the_leader(tmp_path):
    robot = _Robot(fail_on_send=20)
    leader = _FeedbackLeader(_shifted(START, 30.0))
    steps = {5: _cmd(1, "align", task=TASK, auto_start=False, assist=True)}
    code, lines, leader, meta = _run(tmp_path, robot, steps, leader=leader)
    assert code == 1
    assert any(f[0] == "goal" for f in leader.feedback)
    assert leader.feedback[-1] == ("release",) and leader.disconnected == 1


def test_a_failed_connect_reports_the_error_and_leaves_nothing_connected(tmp_path):
    robot = _Robot()
    stream = _Stream()
    leader = _Leader()
    leader.connect = lambda: (_ for _ in ()).throw(ConnectionError("Timeout waiting for leader host"))
    lines: list[str] = []
    code = run_session(_session_cfg(tmp_path), robot, leader, stream=stream, emit=lines.append)
    assert code == 1
    assert lines == ["ERROR ConnectionError: Timeout waiting for leader host"]
    assert not robot.is_connected and robot.disconnected == 1
