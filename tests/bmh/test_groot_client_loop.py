"""Loop-level tests for ``bmh-groot-client`` - fake robot, fake adapter, no server.

Runs ``_run_control_loop`` (and ``main`` for the robot-disconnect guarantees) against
fakes and checks what reaches the robot: an idle start is still and quiet, a target from
the control file connects and moves, an idle update stops motion at once and discards the
chunk that was still in flight, a failing request holds instead of ending the process,
and a failing startup never leaks a connected (torqued) robot.

The scripted watcher is the test clock: the loop polls it exactly once per tick, so it
delivers targets at fixed ticks and ends the run with ``KeyboardInterrupt`` - which the
loop treats as a normal shutdown. Needs the same optional deps as the worker tests;
skipped otherwise. Run with ``uv run --no-sync --with msgpack-numpy pytest tests/bmh -q``.
"""

import threading
from collections.abc import Callable

import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")
pytest.importorskip("zmq")

import bmh.scripts.bmh_groot_client as script  # noqa: E402
from bmh.groot_client.control import InferenceTarget  # noqa: E402
from bmh.scripts.bmh_groot_client import (  # noqa: E402
    BmhInferenceConfig,
    PolicyConnectError,
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
    """Call `n` (1-based) answers with actions `n * 100 + i`, so chunks are told apart."""

    def __init__(self, fail_on: tuple[int, ...] = (), gates: dict[int, threading.Event] | None = None):
        self.policy = None
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.gates = gates or {}

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        self.calls.append(obs["lang"])
        n = len(self.calls)
        if n in self.gates:
            assert self.gates[n].wait(5), "gate was never opened"
        if n in self.fail_on:
            raise RuntimeError("server gone")
        return [{"j": float(n * 100 + i)} for i in range(HORIZON)]


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


def test_main_starts_idle_from_an_empty_control_file_path(fake_main, tmp_path, monkeypatch):
    robot, adapter, state = fake_main
    monkeypatch.setattr(script, "ControlFileWatcher", lambda _path: _ScriptedWatcher(ticks=10))
    monkeypatch.setattr(_ScriptedWatcher, "read_initial", lambda _self: None, raising=False)
    monkeypatch.setattr(_ScriptedWatcher, "path", tmp_path / "control.json", raising=False)
    main(_cfg(control_file=str(tmp_path / "control.json")))
    assert robot.sent == [] and adapter.calls == [] and state["connect"].calls == []
    assert (robot.connected, robot.disconnected) == (1, 1)
