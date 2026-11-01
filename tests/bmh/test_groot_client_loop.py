"""Loop-level tests for ``bmh-groot-client`` - fake robot, fake adapter, no server.

Runs ``_run_control_loop`` (and ``main`` for the robot-disconnect guarantees) against
fakes and checks what reaches the robot: an idle start is still and quiet, a target from
the control file connects and moves, an idle update stops motion at once and discards the
chunk that was still in flight, a failing request holds instead of ending the process,
and a failing startup never leaks a connected (torqued) robot. A relaxed idle target turns
the torque off on both arms' buses and sends nothing until a Hold or a new task, which
stiffen the arms where they were left. A Home idle target ramps the arms from where they
are to the saved pose, holds it and reports the arrival; a newer target stops the ramp. The policy's feedback
values (``feedback_*``) never reach the robot, and come back as ``BMH_FEEDBACK`` lines
under the seq of the target whose chunk set them. Live loop settings (from a scripted loop
watcher) change the tick period, the refetch point and the horizon cut from the next tick /
chunk, are answered with ``BMH_LOOP`` lines, and a bad document is rejected whole.

The scripted watcher is the test clock: the loop polls it exactly once per tick, so it
delivers targets at fixed ticks and ends the run with ``KeyboardInterrupt`` - which the
loop treats as a normal shutdown. Needs the same optional deps as the worker tests;
skipped otherwise. Run with ``uv run --no-sync --with msgpack-numpy pytest tests/bmh -q``.
"""

import json
import re
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")
pytest.importorskip("zmq")

import bmh.scripts.bmh_groot_client as script  # noqa: E402
from bmh.groot_client.control import (  # noqa: E402
    FEEDBACK_STATUS_PREFIX,
    LOOP_STATUS_PREFIX,
    InferenceTarget,
    LoopSettings,
)
from bmh.scripts.bmh_groot_client import (  # noqa: E402
    BmhInferenceConfig,
    PolicyConnectError,
    _blend_actions,
    _run_control_loop,
    main,
)

HORIZON = 8
TICKS = 60


def _cfg(**overrides) -> BmhInferenceConfig:
    # 200 Hz keeps a 60-tick run at ~0.3 s; blending is off so every action sent can be
    # traced back to the chunk it came from.
    values = {"robot": None, "fps": 200, "action_horizon": HORIZON, "refetch_after": 3, "blend_frames": 0}
    return BmhInferenceConfig(**{**values, **overrides})


def _target(seq, host="h1", lang="task A") -> InferenceTarget:
    return InferenceTarget(seq, host, 5555, lang, "t1")


# --------------------------------------------------------------------------- fakes


class _FakeRobot:
    name = "fake"
    action_features = {"j": float}

    def __init__(self):
        self.sent: list[dict[str, float]] = []
        self.observations = 0
        self.connected = 0
        self.disconnected = 0

    def connect(self) -> None:
        self.connected += 1

    def get_observation(self) -> dict:
        self.observations += 1
        return {}

    def send_action(self, action: dict[str, float]) -> None:
        self.sent.append(action)

    def disconnect(self) -> None:
        self.disconnected += 1


class _FakeBus:
    """One arm's motor bus: records torque and position transfers into the robot's `calls`."""

    def __init__(self, arm: str, calls: list[tuple], present: dict[str, float]):
        self.arm = arm
        self.calls = calls
        self.present = present

    def disable_torque(self, num_retry: int = 0) -> None:
        self.calls.append((self.arm, "torque off"))

    def enable_torque(self, num_retry: int = 0) -> None:
        self.calls.append((self.arm, "torque on"))

    def sync_read(self, data_name: str, num_retry: int = 0) -> dict[str, float]:
        self.calls.append((self.arm, f"read {data_name}"))
        return dict(self.present)

    def sync_write(self, data_name: str, values: dict[str, float], num_retry: int = 0) -> None:
        self.calls.append((self.arm, f"write {data_name}", dict(values)))


class _FakeBiRobot(_FakeRobot):
    """`_FakeRobot` with a `bi_so_follower`'s two arms, whose buses can be relaxed.

    `calls` interleaves the bus calls with every observation and send, in order. The
    arms "were moved by hand" to `j = 11` (left) and `j = 22` (right).
    """

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []
        self.left_arm = SimpleNamespace(bus=_FakeBus("left", self.calls, {"j": 11.0}))
        self.right_arm = SimpleNamespace(bus=_FakeBus("right", self.calls, {"j": 22.0}))

    def get_observation(self) -> dict:
        self.calls.append(("robot", "observe"))
        return super().get_observation()

    def send_action(self, action: dict[str, float]) -> None:
        self.calls.append(("robot", "send"))
        super().send_action(action)

    def bus_calls(self) -> list[tuple]:
        return [c for c in self.calls if c[0] != "robot"]


RELAX_CALLS = [("left", "torque off"), ("right", "torque off")]
STIFFEN_CALLS = [
    ("left", "read Present_Position"),
    ("left", "write Goal_Position", {"j": 11.0}),  # pinned before the torque comes on
    ("left", "torque on"),
    ("right", "read Present_Position"),
    ("right", "write Goal_Position", {"j": 22.0}),
    ("right", "torque on"),
]
LEFT_AT_HAND = {"left_j.pos": 11.0, "right_j.pos": 22.0}


class _FakeClient:
    def __init__(self, target: InferenceTarget):
        self.target = target
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Connect:
    """`connect` callable: records the targets it was asked for, optionally failing."""

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[InferenceTarget] = []
        self.clients: list[_FakeClient] = []

    def __call__(self, target: InferenceTarget) -> _FakeClient:
        self.calls.append(target)
        if self.error is not None:
            raise self.error
        self.clients.append(_FakeClient(target))
        return self.clients[-1]


class _FakeAdapter:
    """Call `n` (1-based) answers with actions `n * 100 + i`, so chunks are told apart.

    `feedback(n, i)` adds step `i`'s prefixed `feedback_*` values to call `n`'s chunk, the
    way `BiSoBimanualAdapter.get_action` hands them over.
    """

    def __init__(
        self,
        fail_on: tuple[int, ...] = (),
        gates: dict[int, threading.Event] | None = None,
        feedback: Callable[[int, int], dict[str, float]] | None = None,
        horizon: int = HORIZON,
    ):
        self.policy = None
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.gates = gates or {}
        self.feedback = feedback
        self.horizon = horizon

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        self.calls.append(obs["lang"])
        n = len(self.calls)
        if n in self.gates:
            assert self.gates[n].wait(5), "gate was never opened"
        if n in self.fail_on:
            raise RuntimeError("server gone")
        return [
            {"j": float(n * 100 + i), **(self.feedback(n, i) if self.feedback else {})}
            for i in range(self.horizon)
        ]


class _ScriptedWatcher:
    """Test clock: `poll()` runs once per tick, fires that tick's step, stops after `ticks`."""

    def __init__(self, steps: dict[int, Callable[[], InferenceTarget | None]] | None = None, ticks=TICKS):
        self.steps = steps or {}
        self.ticks = ticks
        self.tick = 0

    def poll(self) -> InferenceTarget | None:
        self.tick += 1
        if self.tick > self.ticks:
            raise KeyboardInterrupt
        step = self.steps.get(self.tick)
        return step() if step else None


def _statuses(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("BMH_TARGET")]


def _feedback_lines(caplog) -> list[dict]:
    """Decoded `BMH_FEEDBACK` payloads, in log order."""
    messages = [r.getMessage() for r in caplog.records]
    return [
        json.loads(m[len(FEEDBACK_STATUS_PREFIX) :]) for m in messages if m.startswith(FEEDBACK_STATUS_PREFIX)
    ]


def _line(seq: int, done: bool | None, value: float | None = None) -> dict:
    """Expected payload for the single feedback `done`; `done=None` is the cleared snapshot."""
    if done is None:
        return {"seq": seq, "flags": {}, "values": {}}
    return {"seq": seq, "flags": {"done": done}, "values": {"done": value}}


def _joints_only(robot: _FakeRobot) -> bool:
    return all(set(action) == {"j"} for action in robot.sent)


# --------------------------------------------------------------------------- loop


def test_idle_start_is_still_and_quiet(caplog):
    robot, adapter, connect = _FakeRobot(), _FakeAdapter(), _Connect()
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, _ScriptedWatcher(), InferenceTarget.make_idle(0), _cfg(), connect)
    assert robot.sent == []
    assert robot.observations == 0
    assert adapter.calls == [] and connect.calls == []
    assert len(_statuses(caplog)) == 1
    assert '"seq": 0, "accepted": true, "error": null, "idle": true' in _statuses(caplog)[0]
    # One line per event, nothing per tick.
    assert len(caplog.records) <= 4


def test_control_file_target_connects_and_moves(caplog):
    robot, adapter, connect = _FakeRobot(), _FakeAdapter(), _Connect()
    watcher = _ScriptedWatcher({5: lambda: _target(1)})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(), connect)
    assert connect.calls == [_target(1)]
    assert adapter.calls and set(adapter.calls) == {"task A"}
    assert robot.sent[0] == {"j": 100.0}  # the first chunk, from its first action
    assert len(robot.sent) > HORIZON  # ...and the loop kept re-fetching
    assert ['"seq": 0, "accepted": true' in s for s in _statuses(caplog)] == [True, False]
    assert '"seq": 1, "accepted": true, "error": null, "idle": false' in _statuses(caplog)[1]
    assert connect.clients[0].closed == 1  # closed at shutdown


def test_idle_update_stops_motion_and_discards_stale_chunk(caplog):
    # Request 2 (the first one the worker serves) hangs until tick 9; the idle target
    # arrives at tick 6, while it is in flight.
    gate = threading.Event()
    robot, adapter, connect = _FakeRobot(), _FakeAdapter(gates={2: gate}), _Connect()
    sent_at_idle: list[int] = []

    def go_idle() -> InferenceTarget:
        sent_at_idle.append(len(robot.sent))
        return InferenceTarget.make_idle(2)

    watcher = _ScriptedWatcher({6: go_idle, 9: gate.set})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(), connect)

    moved = robot.sent[: sent_at_idle[0]]
    assert moved == [{"j": 100.0 + i} for i in range(5)]  # ticks 1-5 played the bootstrap chunk
    assert set(map(str, robot.sent[sent_at_idle[0] :])) == {str(moved[-1])}  # then: hold that pose
    assert all(a["j"] < 200 for a in robot.sent)  # chunk 2 arrived after idle and was dropped
    assert adapter.calls == ["task A"] * 2  # nothing was requested while idle
    assert adapter.policy is None and connect.clients[0].closed == 1
    assert '"seq": 2, "accepted": true, "error": null, "idle": true' in _statuses(caplog)[-1]
    assert "late chunk" not in caplog.text


def test_request_failure_holds_then_new_seq_recovers(caplog):
    robot, adapter, connect = _FakeRobot(), _FakeAdapter(fail_on=(2,)), _Connect()
    sent_at_retry: list[int] = []

    def retry() -> InferenceTarget:
        sent_at_retry.append(len(robot.sent))
        return _target(2, lang="task B")

    watcher = _ScriptedWatcher({30: retry})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(), connect)  # returns: still alive

    failed = [s for s in _statuses(caplog) if '"accepted": false' in s]
    assert len(failed) == 1
    assert '"seq": 1' in failed[0] and "policy request failed: RuntimeError: server gone" in failed[0]
    # Until the retry only the bootstrap chunk was played, then its last reached pose held.
    before_retry = robot.sent[: sent_at_retry[0]]
    assert all(a["j"] < 200 for a in before_retry)
    assert before_retry[-1] == before_retry[-2]
    # The dead server was not asked again; the new seq re-probed the same endpoint and moved on.
    assert adapter.calls[:3] == ["task A", "task A", "task B"]
    assert [t.seq for t in connect.calls] == [1, 2]
    assert connect.clients[0].closed == 1
    assert any(a["j"] >= 300 for a in robot.sent[sent_at_retry[0] :])
    assert caplog.text.count("late chunk") <= 2  # rate-limited, never per tick


def _status_json(line: str) -> dict:
    return json.loads(line[len("BMH_TARGET ") :])


def test_relax_mid_run_goes_limp_and_sends_nothing(caplog):
    # Request 2 hangs until tick 9; the relaxed idle target arrives at tick 6, while it is
    # in flight - its chunk must be dropped like for a plain idle, and nothing re-sent.
    gate = threading.Event()
    robot, adapter, connect = _FakeBiRobot(), _FakeAdapter(gates={2: gate}), _Connect()
    sent_at_relax: list[int] = []

    def relax() -> InferenceTarget:
        sent_at_relax.append(len(robot.sent))
        return InferenceTarget.make_idle(2, relaxed=True)

    watcher = _ScriptedWatcher({6: relax, 9: gate.set})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(), connect)

    assert sent_at_relax == [5]  # ticks 1-5 played the bootstrap chunk
    assert len(robot.sent) == 5  # ...and from the relax on: nothing, not even the held pose
    assert robot.bus_calls() == RELAX_CALLS  # once per bus, never stiffened again
    assert adapter.calls == ["task A"] * 2  # the in-flight chunk landed and was dropped
    last = _status_json(_statuses(caplog)[-1])
    assert (last["seq"], last["accepted"], last["idle"], last["relaxed"]) == (2, True, True, True)
    assert "late chunk" not in caplog.text


def test_relaxed_start_goes_limp(caplog):
    robot, adapter, connect = _FakeBiRobot(), _FakeAdapter(), _Connect()
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, _ScriptedWatcher(), InferenceTarget.make_idle(0, relaxed=True), _cfg(), connect
        )
    assert robot.bus_calls() == RELAX_CALLS
    assert robot.sent == [] and robot.observations == 0
    assert _status_json(_statuses(caplog)[0])["relaxed"] is True


def test_hold_after_relax_stiffens_where_the_arms_were_left():
    robot, adapter, connect = _FakeBiRobot(), _FakeAdapter(), _Connect()
    watcher = _ScriptedWatcher(
        {3: lambda: InferenceTarget.make_idle(1, relaxed=True), 6: lambda: InferenceTarget.make_idle(2)}
    )
    _run_control_loop(robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(), connect)

    assert robot.bus_calls() == RELAX_CALLS + STIFFEN_CALLS
    # Held from the Hold's tick on, every tick, at the pose read back from the buses.
    assert robot.sent == [LEFT_AT_HAND] * (TICKS - 5)
    assert connect.calls == [] and adapter.calls == []


def test_new_task_after_relax_stiffens_before_it_observes():
    robot, adapter, connect = _FakeBiRobot(), _FakeAdapter(), _Connect()
    watcher = _ScriptedWatcher({3: lambda: InferenceTarget.make_idle(1, relaxed=True), 6: lambda: _target(2)})
    _run_control_loop(robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(), connect)

    assert robot.bus_calls() == RELAX_CALLS + STIFFEN_CALLS
    # The new task's first observation sees the arms with the torque already back on.
    first_observe = robot.calls.index(("robot", "observe"))
    assert robot.calls[:first_observe] == RELAX_CALLS + STIFFEN_CALLS + [("robot", "send")]
    assert robot.sent[0] == LEFT_AT_HAND  # held at the hand-placed pose until the chunk lands
    assert connect.calls == [_target(2)]
    assert {"j": 100.0} in robot.sent  # ...then the task runs


# --------------------------------------------------------------------------- home position

HOME = {"left_j.pos": 51.0, "right_j.pos": -18.0}


class _HomeRobot(_FakeBiRobot):
    """`_FakeBiRobot` whose action keys match its buses (`left_j.pos` / `right_j.pos`)."""

    action_features = dict.fromkeys(LEFT_AT_HAND, float)


def _home_statuses(caplog, seq: int) -> list[bool]:
    """`home_reached` of every `BMH_TARGET` line for `seq`, in log order."""
    lines = [_status_json(s) for s in _statuses(caplog)]
    assert all(line["home"] for line in lines if line["seq"] == seq)
    return [line["home_reached"] for line in lines if line["seq"] == seq]


def _monotonic(values: list[float]) -> bool:
    return all(b >= a for a, b in zip(values, values[1:], strict=False)) or all(
        b <= a for a, b in zip(values, values[1:], strict=False)
    )


def test_home_ramps_from_the_present_pose_then_holds_and_reports_arrival(caplog):
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(), _Connect()
    watcher = _ScriptedWatcher({3: lambda: InferenceTarget.make_idle(1, home=HOME)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(home_ramp_s=0.05), connect
        )

    # Nothing was sent before; the ramp starts from the pose read off the buses, torque untouched.
    assert robot.bus_calls() == [("left", "read Present_Position"), ("right", "read Present_Position")]
    assert robot.sent[0] == LEFT_AT_HAND
    for key in HOME:
        assert _monotonic([a[key] for a in robot.sent])
    assert robot.sent[-1] == HOME
    assert robot.sent.count(HOME) > 10  # ...and held there every tick after the arrival
    assert len(robot.sent) == TICKS - 2
    assert _home_statuses(caplog, 1) == [False, True]
    assert connect.calls == [] and adapter.calls == []


def test_home_at_startup_ramps_from_the_present_pose(caplog):
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(), _Connect()
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            adapter,
            _ScriptedWatcher(),
            InferenceTarget.make_idle(0, home=HOME),
            _cfg(home_ramp_s=0.05),
            connect,
        )
    assert robot.sent[0] == LEFT_AT_HAND and robot.sent[-1] == HOME
    assert _home_statuses(caplog, 0) == [False, True]


def test_home_after_relax_stiffens_first_and_starts_from_the_hand_pose(caplog):
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(), _Connect()
    watcher = _ScriptedWatcher(
        {
            3: lambda: InferenceTarget.make_idle(1, relaxed=True),
            6: lambda: InferenceTarget.make_idle(2, home=HOME),
        }
    )
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(home_ramp_s=0.05), connect
        )

    # Stiffened where the operator left the arms; no second read - the ramp starts there.
    assert robot.bus_calls() == RELAX_CALLS + STIFFEN_CALLS
    assert robot.sent[0] == LEFT_AT_HAND
    assert robot.sent[-1] == HOME
    assert _home_statuses(caplog, 2) == [False, True]


def test_new_target_mid_ramp_holds_where_the_robot_is(caplog):
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(), _Connect()
    sent_at_hold: list[int] = []

    def hold() -> InferenceTarget:
        sent_at_hold.append(len(robot.sent))
        return InferenceTarget.make_idle(2)

    watcher = _ScriptedWatcher({3: lambda: InferenceTarget.make_idle(1, home=HOME), 13: hold})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(home_ramp_s=10.0), connect
        )

    here = robot.sent[sent_at_hold[0] - 1]
    assert LEFT_AT_HAND["left_j.pos"] < here["left_j.pos"] < HOME["left_j.pos"]
    assert robot.sent[sent_at_hold[0] :] == [here] * (len(robot.sent) - sent_at_hold[0])
    assert _home_statuses(caplog, 1) == [False]  # never reached


def test_home_arrival_is_reported_after_the_worker_accepted_the_seq(caplog):
    # Request 2 hangs until tick 40 and then fails; Home (seq 2) arrives at tick 5 and the
    # ramp ends long before the worker gets to seq 2. Its failure line for seq 1 must not
    # land between seq 2's accepted and reached lines.
    gate = threading.Event()
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(fail_on=(2,), gates={2: gate}), _Connect()
    watcher = _ScriptedWatcher({5: lambda: InferenceTarget.make_idle(2, home=HOME), 40: lambda: gate.set()})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(home_ramp_s=0.05), connect)

    lines = [_status_json(s) for s in _statuses(caplog)]
    assert [(line["seq"], line["accepted"], line["home_reached"]) for line in lines] == [
        (1, True, False),
        (1, False, False),
        (2, True, False),
        (2, True, True),
    ]


def test_home_keys_the_robot_does_not_have_are_dropped(caplog):
    robot, adapter, connect = _HomeRobot(), _FakeAdapter(), _Connect()
    home = {"left_j.pos": 30.0, "left_head_pan.pos": 5.0}
    watcher = _ScriptedWatcher({3: lambda: InferenceTarget.make_idle(1, home=home)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, watcher, InferenceTarget.make_idle(0), _cfg(home_ramp_s=0.05), connect
        )

    assert "left_head_pan.pos" in caplog.text
    assert all(set(a) == set(LEFT_AT_HAND) for a in robot.sent)
    assert robot.sent[-1] == {"left_j.pos": 30.0, "right_j.pos": 22.0}  # right arm stays put
    assert _home_statuses(caplog, 1) == [False, True]


def test_late_chunk_warning_is_rate_limited(caplog):
    # Request 2 hangs until the last tick: the bootstrap chunk runs dry at tick 9 and the
    # robot waits ~50 ticks (0.25 s) for it.
    gate = threading.Event()
    robot, adapter = _FakeRobot(), _FakeAdapter(gates={2: gate})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, _ScriptedWatcher({TICKS: gate.set}), _target(1), _cfg(), _Connect())
    assert caplog.text.count("late chunk") == 1
    assert len(robot.sent) == TICKS  # every tick of the gap re-sent the last pose


def test_startup_connect_failure_exits_without_a_worker():
    robot, adapter = _FakeRobot(), _FakeAdapter()
    connect = _Connect(PolicyConnectError("Cannot reach GR00T policy server at h1:5555"))
    with pytest.raises(SystemExit, match="Cannot reach"):
        _run_control_loop(robot, adapter, _ScriptedWatcher(), _target(1), _cfg(), connect)
    assert robot.sent == [] and adapter.policy is None
    assert not [t for t in threading.enumerate() if t.name == "bmh-inference-worker"]


# --------------------------------------------------------------------------- feedback


def test_ramping_feedback_reports_true_once_after_the_debounce(caplog):
    # One 24-step chunk whose `feedback_done` ramps i / 16; request 2 hangs until the last
    # tick, so tick t plays step t - 1 and nothing else feeds the tracker. The value is at
    # or above 0.6 from step 10 (0.625) on: the 5th such tick in a row is step 14 (0.875).
    gate = threading.Event()
    robot = _FakeRobot()
    adapter = _FakeAdapter(gates={2: gate}, horizon=24, feedback=lambda _n, i: {"feedback_done": i / 16})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            adapter,
            _ScriptedWatcher({TICKS: gate.set}),
            _target(1),
            _cfg(action_horizon=24),
            _Connect(),
        )
    assert _feedback_lines(caplog) == [_line(1, False, 0.0), _line(1, True, 0.875)]
    # Nothing but joints ever reached the robot — the late-chunk ticks re-send the last
    # *action*, not the last chunk step.
    assert len(robot.sent) == TICKS and _joints_only(robot)
    assert robot.sent[23] == robot.sent[40] == {"j": 123.0}


def test_no_feedback_keys_means_no_feedback_lines(caplog):
    robot = _FakeRobot()
    with caplog.at_level("INFO"):
        _run_control_loop(robot, _FakeAdapter(), _ScriptedWatcher(), _target(1), _cfg(), _Connect())
    assert len(robot.sent) > HORIZON
    assert FEEDBACK_STATUS_PREFIX.strip() not in caplog.text


def test_idle_clears_the_feedback_flags(caplog):
    # `done` is on from tick 5 (default debounce); the idle target arrives at tick 7 while
    # request 2 is still in flight, its stale chunk lands during idle (tick 9+).
    gate = threading.Event()
    robot = _FakeRobot()
    adapter = _FakeAdapter(gates={2: gate}, feedback=lambda _n, _i: {"feedback_done": 1.0})
    lines_before_gate: list[int] = []

    def open_gate() -> None:
        lines_before_gate.append(len(_feedback_lines(caplog)))
        gate.set()

    watcher = _ScriptedWatcher({7: lambda: InferenceTarget.make_idle(2), 9: open_gate})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(), _Connect())
    # Cleared once - under the seq the flags were reported for - not again when the worker's
    # `Idle` reply and the stale chunk arrive.
    assert _feedback_lines(caplog) == [_line(1, False, 1.0), _line(1, True, 1.0), _line(1, None)]
    # ...and at once, with the robot: not when the worker (still busy with the stale
    # request, for up to `timeout_ms` on a slow server) gets round to answering `Idle`.
    assert lines_before_gate == [3]
    assert len(robot.sent) == TICKS and _joints_only(robot)  # incl. ~50 idle hold ticks
    assert robot.sent[-1] == {"j": 105.0}


def test_hold_after_a_failed_request_clears_the_feedback_flags(caplog):
    # Request 2 fails, but only once the gate opens at tick 7 - `done` is on by then.
    gate = threading.Event()
    robot = _FakeRobot()
    adapter = _FakeAdapter(
        fail_on=(2,), gates={2: gate}, feedback=lambda _n, _i: {"feedback_done": 1.0}, horizon=24
    )
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, _ScriptedWatcher({7: gate.set}), _target(1), _cfg(action_horizon=24), _Connect()
        )
    assert _feedback_lines(caplog) == [_line(1, False, 1.0), _line(1, True, 1.0), _line(1, None)]
    assert len(robot.sent) == TICKS and _joints_only(robot)  # incl. the hold ticks
    assert robot.sent[-1] == robot.sent[-2]


def test_inflight_chunk_of_the_old_target_sets_nothing_under_the_new_seq(caplog):
    # Request 2 is fired under target #1 and hangs; target #2 arrives at tick 5; the gate
    # opens at tick 7. Chunk 2 - the only one reporting `done` - is therefore played while
    # target #2 is already current. With a debounce of 1 a single fed step would flip the
    # flag, so any leak shows.
    gate = threading.Event()
    robot = _FakeRobot()
    adapter = _FakeAdapter(gates={2: gate}, feedback=lambda n, _i: {"feedback_done": 1.0 if n == 2 else 0.0})
    watcher = _ScriptedWatcher({5: lambda: _target(2, lang="task B"), 7: gate.set})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(feedback_debounce_ticks=1), _Connect())
    assert adapter.calls[:3] == ["task A", "task A", "task B"]
    assert any(200 <= a["j"] < 300 for a in robot.sent)  # the old target's chunk did play
    # Never on: not under #1 (no longer current), not under #2 (not its chunk). #2's own
    # first chunk then re-announces the flag under the new seq.
    assert _feedback_lines(caplog) == [_line(1, False, 0.0), _line(2, False, 0.0)]
    assert _joints_only(robot)


def test_new_target_starts_its_flags_from_scratch(caplog):
    # Every chunk says `done`; a prompt-only change arrives at tick 20. The flag must not
    # carry over: target #2 announces it off, then debounces it on from its own chunks.
    robot = _FakeRobot()
    adapter = _FakeAdapter(feedback=lambda _n, _i: {"feedback_done": 1.0})
    watcher = _ScriptedWatcher({20: lambda: _target(2, lang="task B")})
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, watcher, _target(1), _cfg(), _Connect())
    assert _feedback_lines(caplog) == [
        _line(1, False, 1.0),
        _line(1, True, 1.0),
        _line(2, False, 1.0),
        _line(2, True, 1.0),
    ]
    assert _joints_only(robot)


def test_blending_chunks_with_different_feedback_keys_does_not_raise(caplog):
    # Consecutive chunks disagree on their feedback keys (as across a skill switch):
    # none / {a} / {a, b}, crossfaded over 3 frames at every swap.
    def feedback(n: int, _i: int) -> dict[str, float]:
        return [{}, {"feedback_a": 1.0}, {"feedback_a": 0.0, "feedback_b": 1.0}][n % 3]

    robot = _FakeRobot()
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            _FakeAdapter(feedback=feedback),
            _ScriptedWatcher(),
            _target(1),
            _cfg(blend_frames=3),
            _Connect(),
        )
    blended = [int(n) for n in re.findall(r"(\d+) blended", caplog.text)]
    assert blended and max(blended) > 0  # the crossfade really ran
    assert len(robot.sent) > HORIZON and _joints_only(robot)


def test_blend_actions_takes_feedback_from_the_new_chunk_unblended():
    old = {"j": 0.0, "feedback_done": 1.0, "feedback_old_only": 1.0}
    new = {"j": 10.0, "feedback_done": 0.0, "feedback_new_only": 0.5}
    assert _blend_actions(old, new, 0.25) == {"j": 2.5, "feedback_done": 0.0, "feedback_new_only": 0.5}


def test_feedback_config_defaults():
    cfg = BmhInferenceConfig(robot=None)
    assert (cfg.feedback_on_threshold, cfg.feedback_off_threshold, cfg.feedback_debounce_ticks) == (
        0.6,
        0.4,
        5,
    )


# --------------------------------------------------------------------------- loop settings


class _ScriptedLoopWatcher:
    """Loop-file stand-in: `poll()` runs once per tick, after the target watcher, and fires
    that tick's step. Never ends the run - the target watcher is the clock."""

    def __init__(self, steps: dict[int, Callable[[], LoopSettings | None]] | None = None):
        self.steps = steps or {}
        self.tick = 0

    def poll(self) -> LoopSettings | None:
        self.tick += 1
        step = self.steps.get(self.tick)
        return step() if step else None


class _VirtualClock:
    """Stands in for the script's `time` module: `perf_counter` only advances in `sleep`.

    A tick then takes no time, so every sleep is exactly the tick's period.
    """

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _settings(seq: int, horizon: int = HORIZON, refetch: int = 3, speed: int = 100) -> LoopSettings:
    return LoopSettings(seq, horizon, refetch, speed)


def _loop_lines(caplog) -> list[dict]:
    """Decoded `BMH_LOOP` payloads, in log order."""
    messages = [r.getMessage() for r in caplog.records]
    return [json.loads(m[len(LOOP_STATUS_PREFIX) :]) for m in messages if m.startswith(LOOP_STATUS_PREFIX)]


def _loop_line(seq: int, accepted: bool, horizon: int, refetch: int, speed: int, error=None) -> dict:
    return {
        "seq": seq,
        "accepted": accepted,
        "error": error,
        "action_horizon": horizon,
        "refetch_after": refetch,
        "speed_pct": speed,
    }


def _fires(caplog) -> list[tuple[int, int]]:
    """`(frame, consumed_since_swap)` of every fired request."""
    found = re.findall(r"inference fired at frame=(\d+) \(consumed_since_swap=(\d+)\)", caplog.text)
    return [(int(frame), int(consumed)) for frame, consumed in found]


def _arrivals(caplog) -> list[int]:
    """Actions per swapped-in chunk, after the horizon cut."""
    return [int(n) for n in re.findall(r"chunk swap: (\d+) actions arrived", caplog.text)]


@pytest.mark.parametrize(("speed", "period"), [(150, 1 / 300), (50, 1 / 100)])
def test_speed_change_scales_the_tick_period_from_the_next_tick(monkeypatch, caplog, speed, period):
    clock = _VirtualClock()
    monkeypatch.setattr(script, "time", clock)
    robot = _FakeRobot()
    loop = _ScriptedLoopWatcher({21: lambda: _settings(1, speed=speed)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, _FakeAdapter(), _ScriptedWatcher(), _target(1), _cfg(), _Connect(), loop_watcher=loop
        )
    assert clock.sleeps[:20] == pytest.approx([1 / 200] * 20)  # 100 % of fps=200
    assert clock.sleeps[20:] == pytest.approx([period] * (TICKS - 20))
    assert len(robot.sent) == TICKS  # still one action per tick, whatever the speed
    assert _loop_lines(caplog) == [
        _loop_line(0, True, HORIZON, 3, 100),
        _loop_line(1, True, HORIZON, 3, speed),
    ]
    assert f"loop settings #1 applied: horizon={HORIZON} refetch_after=3 speed={speed}%" in caplog.text


def test_refetch_change_moves_the_next_fire(caplog):
    robot = _FakeRobot()
    loop = _ScriptedLoopWatcher({20: lambda: _settings(1, refetch=6)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, _FakeAdapter(), _ScriptedWatcher(), _target(1), _cfg(), _Connect(), loop_watcher=loop
        )
    fires = _fires(caplog)
    assert {consumed for frame, consumed in fires if frame < 20} == {3}
    after = [consumed for frame, consumed in fires if frame > 20]
    assert after and set(after) == {6}
    assert "late chunk" not in caplog.text


def test_horizon_is_cut_on_arrival_including_the_request_in_flight(caplog):
    # The policy answers 24 steps; the loop keeps 8. Request 3 is fired under horizon 8 and
    # hangs until tick 20; the horizon drops to 4 at tick 15, while it is in flight.
    gate = threading.Event()
    robot, adapter = _FakeRobot(), _FakeAdapter(gates={3: gate}, horizon=24)
    loop = _ScriptedLoopWatcher({15: lambda: _settings(1, horizon=4)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            adapter,
            _ScriptedWatcher({20: gate.set}),
            _target(1),
            _cfg(),
            _Connect(),
            loop_watcher=loop,
        )
    arrivals = _arrivals(caplog)
    assert arrivals[0] == 8  # chunk 2, under horizon 8
    assert len(arrivals) > 2 and set(arrivals[1:]) == {4}  # chunk 3 (in flight at the change) on
    # Chunk 2 played out to its 8th step and the robot waited; chunk 3 only had 4 steps.
    played = [a["j"] for a in robot.sent]
    assert max(j for j in played if 200 <= j < 300) == 207
    assert max(j for j in played if 300 <= j < 400) == 303
    assert "shorter than action_horizon" not in caplog.text


def test_stall_guard_fires_at_chunk_end_when_refetch_is_raised_above_what_is_left(caplog):
    # Request 2 fires at tick 3 and is held until tick 6, so the swap drops 3 stale actions and
    # keeps 5. Then refetch_after goes to 7: the counter can never reach it on this chunk.
    gate = threading.Event()
    robot, adapter = _FakeRobot(), _FakeAdapter(gates={2: gate})

    def release() -> None:
        gate.set()
        time.sleep(0.05)  # the chunk is in the queue before this tick's swap

    loop = _ScriptedLoopWatcher({7: lambda: _settings(1, refetch=HORIZON - 1)})
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot, adapter, _ScriptedWatcher({6: release}), _target(1), _cfg(), _Connect(), loop_watcher=loop
        )
    assert "5 kept" in caplog.text
    assert (10, 5) in _fires(caplog)  # fired at the chunk's end instead of holding forever
    assert len(adapter.calls) >= 3 and any(a["j"] >= 300 for a in robot.sent)
    assert len(robot.sent) > HORIZON


def test_stall_guard_keeps_refetching_a_policy_with_short_chunks(caplog):
    robot, adapter = _FakeRobot(), _FakeAdapter(horizon=4)
    with caplog.at_level("INFO"):
        _run_control_loop(robot, adapter, _ScriptedWatcher(), _target(1), _cfg(refetch_after=6), _Connect())
    assert len(adapter.calls) > 5
    assert any(a["j"] >= 500 for a in robot.sent)
    assert caplog.text.count("shorter than action_horizon") == 1
    assert (
        "policy returns 4-step chunks, shorter than action_horizon=8 — the loop refetches at chunk end"
        in (caplog.text)
    )


def test_bad_loop_settings_are_rejected_whole_and_change_nothing(monkeypatch, caplog):
    clock = _VirtualClock()
    monkeypatch.setattr(script, "time", clock)
    robot = _FakeRobot()
    loop = _ScriptedLoopWatcher(
        {
            10: lambda: _settings(1, horizon=2, refetch=1, speed=150),  # horizon <= blend_frames
            20: lambda: _settings(2, horizon=HORIZON, refetch=HORIZON, speed=150),  # refetch >= horizon
            30: lambda: _settings(3, speed=150),  # a good one still applies afterwards
        }
    )
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            _FakeAdapter(),
            _ScriptedWatcher(),
            _target(1),
            _cfg(blend_frames=2),
            _Connect(),
            loop_watcher=loop,
        )
    lines = _loop_lines(caplog)
    assert [(line["seq"], line["accepted"]) for line in lines] == [
        (0, True),
        (1, False),
        (2, False),
        (3, True),
    ]
    # A rejection reports the rejected seq with the values still in effect.
    assert lines[1] == _loop_line(
        1, False, HORIZON, 3, 100, error="action_horizon must be greater than blend_frames=2, got 2"
    )
    assert lines[2]["error"].startswith("refetch_after must be in [1, action_horizon=8)")
    assert (lines[2]["action_horizon"], lines[2]["refetch_after"], lines[2]["speed_pct"]) == (HORIZON, 3, 100)
    assert clock.sleeps[:29] == pytest.approx([1 / 200] * 29)  # the 150 % of #1 / #2 never applied
    assert clock.sleeps[29:] == pytest.approx([1 / 300] * (TICKS - 29))
    assert {consumed for _frame, consumed in _fires(caplog)} == {3}
    assert len(robot.sent) == TICKS


def test_startup_reports_the_cli_loop_settings_without_a_loop_file(caplog):
    with caplog.at_level("INFO"):
        _run_control_loop(
            _FakeRobot(), _FakeAdapter(), _ScriptedWatcher(), InferenceTarget.make_idle(0), _cfg(), _Connect()
        )
    assert _loop_lines(caplog) == [_loop_line(0, True, HORIZON, 3, 100)]
    # ...right after the startup target line.
    messages = [r.getMessage() for r in caplog.records]
    first_target = next(i for i, m in enumerate(messages) if m.startswith("BMH_TARGET"))
    assert messages[first_target + 1].startswith(LOOP_STATUS_PREFIX)


def test_startup_reports_the_given_loop_settings_once(caplog):
    robot, adapter = _FakeRobot(), _FakeAdapter(horizon=24)
    with caplog.at_level("INFO"):
        _run_control_loop(
            robot,
            adapter,
            _ScriptedWatcher(),
            _target(1),
            _cfg(),
            _Connect(),
            loop_watcher=_ScriptedLoopWatcher(),
            settings=LoopSettings(5, 6, 2, 80),
        )
    assert _loop_lines(caplog) == [_loop_line(5, True, 6, 2, 80)]
    assert set(_arrivals(caplog)) == {6}  # 24-step chunks cut to the given horizon
    assert {consumed for _frame, consumed in _fires(caplog)} == {2}


def test_a_real_loop_file_rewritten_mid_run_applies_on_that_tick(tmp_path, monkeypatch, caplog):
    # The app's write: tmp + rename. The target watcher (the clock) does it at tick 10; the loop
    # watcher polls after it in the same tick.
    clock = _VirtualClock()
    monkeypatch.setattr(script, "time", clock)
    path = tmp_path / "inference-loop.json"

    def write(document: dict) -> None:
        tmp = tmp_path / "inference-loop.json.tmp"
        tmp.write_text(json.dumps(document))
        tmp.replace(path)

    write({"seq": 1, "action_horizon": HORIZON, "refetch_after": 3, "speed_pct": 100})
    loop = script.LoopFileWatcher(path)
    settings = loop.read_initial()
    watcher = _ScriptedWatcher(
        {10: lambda: write({"seq": 2, "action_horizon": HORIZON, "refetch_after": 3, "speed_pct": 50})}
    )
    with caplog.at_level("INFO"):
        _run_control_loop(
            _FakeRobot(),
            _FakeAdapter(),
            watcher,
            _target(1),
            _cfg(),
            _Connect(),
            loop_watcher=loop,
            settings=settings,
        )
    assert _loop_lines(caplog) == [_loop_line(1, True, HORIZON, 3, 100), _loop_line(2, True, HORIZON, 3, 50)]
    assert clock.sleeps[:9] == pytest.approx([1 / 200] * 9)
    assert clock.sleeps[9:] == pytest.approx([1 / 100] * (TICKS - 9))


# --------------------------------------------------------------------------- main


@pytest.fixture
def fake_main(monkeypatch):
    """`main` wired to a fake robot / adapter; returns them plus the failing-connect switch."""
    robot, adapter = _FakeRobot(), _FakeAdapter()
    state = {"connect": _Connect()}
    monkeypatch.setattr(script, "init_logging", lambda: None)
    monkeypatch.setattr(script, "make_robot_from_config", lambda _cfg: robot)
    monkeypatch.setattr(script, "BiSoBimanualAdapter", lambda *_a, **_kw: adapter)
    monkeypatch.setattr(script, "_connect_policy", lambda target, *_a: state["connect"](target))
    return robot, adapter, state


def test_main_startup_connect_failure_disconnects_robot(fake_main):
    robot, _adapter, state = fake_main
    state["connect"] = _Connect(PolicyConnectError("Cannot reach GR00T policy server at h1:5555"))
    with pytest.raises(SystemExit, match="Cannot reach"):
        main(_cfg(lang_instruction="task A"))
    assert (robot.connected, robot.disconnected) == (1, 1)


def test_main_bootstrap_failure_disconnects_robot_and_closes_client(fake_main):
    robot, adapter, state = fake_main
    adapter.fail_on = (1,)
    with pytest.raises(RuntimeError, match="server gone"):
        main(_cfg(lang_instruction="task A"))
    assert (robot.connected, robot.disconnected) == (1, 1)
    assert state["connect"].clients[0].closed == 1


def test_main_without_target_or_control_file_never_touches_the_robot(fake_main):
    robot, _adapter, _state = fake_main
    with pytest.raises(SystemExit, match="--control_file"):
        main(_cfg())
    assert robot.connected == 0


@pytest.mark.parametrize(
    "overrides",
    [{"feedback_on_threshold": 0.3}, {"feedback_off_threshold": 0.7}, {"feedback_debounce_ticks": 0}],
)
def test_main_rejects_bad_feedback_settings_before_touching_the_robot(fake_main, overrides):
    robot, _adapter, _state = fake_main
    with pytest.raises(SystemExit, match="--feedback_"):
        main(_cfg(lang_instruction="task A", **overrides))
    assert robot.connected == 0


@pytest.mark.parametrize("home_ramp_s", [0.0, -1.0])
def test_main_rejects_a_non_positive_home_ramp_before_touching_the_robot(fake_main, home_ramp_s):
    robot, _adapter, _state = fake_main
    with pytest.raises(SystemExit, match="--home_ramp_s"):
        main(_cfg(lang_instruction="task A", home_ramp_s=home_ramp_s))
    assert robot.connected == 0


def test_main_starts_idle_from_an_empty_control_file_path(fake_main, tmp_path, monkeypatch):
    robot, adapter, state = fake_main
    monkeypatch.setattr(script, "ControlFileWatcher", lambda _path: _ScriptedWatcher(ticks=10))
    monkeypatch.setattr(_ScriptedWatcher, "read_initial", lambda _self: None, raising=False)
    monkeypatch.setattr(_ScriptedWatcher, "path", tmp_path / "control.json", raising=False)
    main(_cfg(control_file=str(tmp_path / "control.json")))
    assert robot.sent == [] and adapter.calls == [] and state["connect"].calls == []
    assert (robot.connected, robot.disconnected) == (1, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"speed_pct": 49},
        {"speed_pct": 151},
        {"refetch_after": HORIZON},
        {"refetch_after": 0},
        {"blend_frames": HORIZON},
        {"blend_frames": -1},
    ],
)
def test_main_rejects_bad_loop_flags_before_touching_the_robot(fake_main, overrides):
    robot, _adapter, _state = fake_main
    with pytest.raises(SystemExit, match=r"Invalid loop settings \(CLI flags\)"):
        main(_cfg(lang_instruction="task A", **overrides))
    assert robot.connected == 0


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ("{not json", "--loop_file"),
        ({"seq": 1, "action_horizon": 8, "refetch_after": 8, "speed_pct": 100}, "--loop_file.*refetch_after"),
        ({"seq": 1, "action_horizon": 8, "refetch_after": 3, "speed_pct": 100.0}, "--loop_file.*speed_pct"),
        # Valid for the parser, but not above this client's --blend_frames=3.
        (
            {"seq": 1, "action_horizon": 3, "refetch_after": 2, "speed_pct": 100},
            r"Invalid loop settings \(--loop_file .*blend_frames=3",
        ),
    ],
)
def test_main_rejects_an_invalid_loop_file_before_touching_the_robot(fake_main, tmp_path, document, match):
    robot, _adapter, _state = fake_main
    path = tmp_path / "loop.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    with pytest.raises(SystemExit, match=match):
        main(_cfg(lang_instruction="task A", blend_frames=3, loop_file=str(path)))
    assert robot.connected == 0


def test_main_starts_with_the_loop_file_over_the_cli_flags(fake_main, tmp_path, monkeypatch, caplog):
    robot, _adapter, _state = fake_main
    monkeypatch.setattr(script, "ControlFileWatcher", lambda _path: _ScriptedWatcher(ticks=10))
    monkeypatch.setattr(_ScriptedWatcher, "read_initial", lambda _self: None, raising=False)
    monkeypatch.setattr(_ScriptedWatcher, "path", tmp_path / "control.json", raising=False)
    path = tmp_path / "loop.json"
    path.write_text(json.dumps({"seq": 7, "action_horizon": 12, "refetch_after": 4, "speed_pct": 120}))
    with caplog.at_level("INFO"):
        main(_cfg(control_file=str(tmp_path / "control.json"), loop_file=str(path), speed_pct=60))
    assert _loop_lines(caplog) == [_loop_line(7, True, 12, 4, 120)]
    assert (robot.connected, robot.disconnected) == (1, 1)


def test_main_without_a_loop_file_starts_from_the_cli_flags(fake_main, tmp_path, monkeypatch, caplog):
    _robot, _adapter, _state = fake_main
    monkeypatch.setattr(script, "ControlFileWatcher", lambda _path: _ScriptedWatcher(ticks=10))
    monkeypatch.setattr(_ScriptedWatcher, "read_initial", lambda _self: None, raising=False)
    monkeypatch.setattr(_ScriptedWatcher, "path", tmp_path / "control.json", raising=False)
    with caplog.at_level("INFO"):
        main(
            _cfg(
                control_file=str(tmp_path / "control.json"),
                loop_file=str(tmp_path / "missing.json"),
                speed_pct=60,
            )
        )
    assert _loop_lines(caplog) == [_loop_line(0, True, HORIZON, 3, 60)]
    assert "missing.json does not exist yet" in caplog.text
