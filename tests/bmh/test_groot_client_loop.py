"""Loop-level tests for ``bmh-groot-client`` - fake robot, fake adapter, no server.

Runs ``_run_control_loop`` (and ``main`` for the robot-disconnect guarantees) against
fakes and checks what reaches the robot: an idle start is still and quiet, a target from
the control file connects and moves, an idle update stops motion at once and discards the
chunk that was still in flight, a failing request holds instead of ending the process,
and a failing startup never leaks a connected (torqued) robot. The policy's feedback
values (``feedback_*``) never reach the robot, and come back as ``BMH_FEEDBACK`` lines
under the seq of the target whose chunk set them.

The scripted watcher is the test clock: the loop polls it exactly once per tick, so it
delivers targets at fixed ticks and ends the run with ``KeyboardInterrupt`` - which the
loop treats as a normal shutdown. Needs the same optional deps as the worker tests;
skipped otherwise. Run with ``uv run --no-sync --with msgpack-numpy pytest tests/bmh -q``.
"""

import json
import re
import threading
from collections.abc import Callable

import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")
pytest.importorskip("zmq")

import bmh.scripts.bmh_groot_client as script  # noqa: E402
from bmh.groot_client.control import FEEDBACK_STATUS_PREFIX, InferenceTarget  # noqa: E402
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


def test_main_starts_idle_from_an_empty_control_file_path(fake_main, tmp_path, monkeypatch):
    robot, adapter, state = fake_main
    monkeypatch.setattr(script, "ControlFileWatcher", lambda _path: _ScriptedWatcher(ticks=10))
    monkeypatch.setattr(_ScriptedWatcher, "read_initial", lambda _self: None, raising=False)
    monkeypatch.setattr(_ScriptedWatcher, "path", tmp_path / "control.json", raising=False)
    main(_cfg(control_file=str(tmp_path / "control.json")))
    assert robot.sent == [] and adapter.calls == [] and state["connect"].calls == []
    assert (robot.connected, robot.disconnected) == (1, 1)
