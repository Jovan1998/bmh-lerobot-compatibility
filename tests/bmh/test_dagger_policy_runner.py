"""`PolicyRunner` - the policy client's chunk loop as a class, against a fake adapter.

Ports the chunk cases of ``test_policy_client_loop.py`` / ``test_policy_client_worker.py``
(time-aligned stale drop and its two bounds, seam blend, refetch stall guard, late chunk,
a rejected or lost server, an instruction-only switch) and adds what a DAgger session
needs on top: `suspend` drops a reply that is still in flight, `resume` fires at once and
plays nothing before a chunk requested from there arrives, and a target set while
suspended is probed without moving the robot.

Every test drives the runner by hand, one `tick` + `fire` per step, and waits for the
worker's reply where the order matters, so nothing depends on wall-clock timing. Needs
the same optional deps as the loop tests; skipped otherwise.
"""

import threading
import time
from collections.abc import Callable

import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")
pytest.importorskip("zmq")

from bmh.dagger.policy_runner import PolicyRunner  # noqa: E402
from bmh.policy_client.control import InferenceTarget, LoopSettings  # noqa: E402
from bmh.scripts.bmh_policy_client import PolicyConnectError  # noqa: E402

HORIZON = 8
FPS = 30
POSE = {"j": 0.0}


def _settings(seq=0, horizon=HORIZON, refetch=3, speed=100, blend=0, lookahead=0, smoothing=0):
    return LoopSettings(
        seq=seq,
        action_horizon=horizon,
        refetch_after=refetch,
        speed_pct=speed,
        blend_frames=blend,
        lookahead_frames=lookahead,
        smoothing_pct=smoothing,
    )


class _FakeClient:
    def __init__(self, target: InferenceTarget):
        self.target = target
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _Connect:
    """`connect` callable: records the targets it was asked for; fails while `error` is set."""

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

    def __init__(
        self,
        fail_on: tuple[int, ...] = (),
        gates: dict[int, threading.Event] | None = None,
        horizon: Callable[[int], int] | int = HORIZON,
        feedback: bool = False,
    ):
        self.policy = None
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.gates = gates or {}
        self.horizon = horizon
        self.feedback = feedback

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        self.calls.append(obs["lang"])
        n = len(self.calls)
        if n in self.gates:
            assert self.gates[n].wait(5), "gate was never opened"
        if n in self.fail_on:
            raise RuntimeError("server gone")
        horizon = self.horizon(n) if callable(self.horizon) else self.horizon
        extra = {"feedback_done": 1.0} if self.feedback else {}
        return [{"j": float(n * 100 + i), **extra} for i in range(horizon)]


class _Rig:
    """A runner with its fakes; `step()` is one control tick (tick, then fire)."""

    def __init__(self, adapter: _FakeAdapter | None = None, connect: _Connect | None = None, **settings):
        self.adapter = adapter or _FakeAdapter()
        self.connect = connect or _Connect()
        self.runner = PolicyRunner(self.adapter, self.connect, _settings(**settings), FPS)
        self.now = 0.0
        self.observations = 0
        self.played: list[dict[str, float] | None] = []

    def observe(self) -> dict:
        self.observations += 1
        return {}

    def step(self) -> dict[str, float] | None:
        self.now += self.runner.period_s()
        action = self.runner.tick(self.now)
        self.played.append(action)
        self.runner.fire(self.observe)
        return action

    def wait_reply(self) -> None:
        """Block until the worker's answer to the request in flight is on the queue."""
        assert self.runner.inflight, "nothing is in flight"
        deadline = time.monotonic() + 5
        while self.runner._chunk_queue.empty():
            assert time.monotonic() < deadline, "the worker never answered"
            time.sleep(0.001)

    def step_with_reply(self) -> dict[str, float] | None:
        self.wait_reply()
        return self.step()

    def start(self, lang: str = "task A") -> None:
        """Target set and resumed; the first chunk (call 1) is swapped in by the next `step`."""
        self.runner.set_target("h1", 5555, lang, "t1")
        self.runner.resume(dict(POSE))
        assert self.step() is None  # nothing to play yet; this tick fired request 1
        self.wait_reply()

    def values(self) -> list[float | None]:
        return [None if a is None else a["j"] for a in self.played]

    def close(self) -> None:
        self.runner.close()


@pytest.fixture
def rig_factory():
    rigs: list[_Rig] = []

    def make(*args, **kwargs) -> _Rig:
        rigs.append(_Rig(*args, **kwargs))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


# --------------------------------------------------------------------------- start / state


def test_no_target_means_nothing_fires(rig_factory):
    rig = rig_factory()
    assert rig.runner.policy_state == "none"
    assert rig.step() is None
    assert rig.observations == 0 and not rig.runner.inflight
    with pytest.raises(RuntimeError, match="no policy target"):
        rig.runner.resume(dict(POSE))


def test_resume_fires_at_once_and_holds_until_the_first_chunk(rig_factory, caplog):
    rig = rig_factory()
    with caplog.at_level("INFO"):
        rig.runner.set_target("h1", 5555, "task A", "t1")
        assert rig.runner.policy_state == "connecting"
        rig.runner.resume(dict(POSE))
        assert rig.step() is None  # no synchronous bootstrap: the caller holds
        assert rig.observations == 1 and rig.runner.inflight
        assert rig.step_with_reply() == {"j": 100.0}  # nothing dropped: the robot was holding
    assert rig.runner.policy_state == "ready"
    assert rig.connect.calls[0].lang_instruction == "task A"
    assert rig.connect.calls[0].api_token == "t1"
    accepted = [r.getMessage() for r in caplog.records if r.getMessage().startswith("BMH_TARGET")]
    assert len(accepted) == 1 and '"seq": 1, "accepted": true' in accepted[0]
    assert "t1" not in caplog.text  # the token is never logged


def test_a_target_set_while_suspended_is_probed_without_playing(rig_factory):
    rig = rig_factory()
    rig.runner.set_target("h1", 5555, "task A", "")
    assert rig.step() is None  # fired the probe
    assert rig.observations == 1
    assert rig.step_with_reply() is None
    assert rig.runner.policy_state == "ready"
    assert rig.runner.current_chunk == []
    # Nothing else is requested while suspended.
    for _ in range(5):
        assert rig.step() is None
    assert rig.adapter.calls == ["task A"]


def test_feedback_values_never_reach_the_action(rig_factory):
    rig = rig_factory(_FakeAdapter(feedback=True))
    rig.start()
    assert rig.step() == {"j": 100.0}


# --------------------------------------------------------------------------- swap


def test_stale_actions_are_dropped_by_the_frames_the_request_was_in_flight(rig_factory):
    gate = threading.Event()
    rig = rig_factory(_FakeAdapter(gates={2: gate}))
    rig.start()
    # 100, 101, 102: the third fires request 2 (5 actions left), which hangs for 2 ticks.
    assert [rig.step()["j"] for _ in range(5)] == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert rig.runner.remaining_at_fire == 5
    gate.set()
    # Arrived 3 frames after it fired: its first 3 actions are where the robot already was.
    assert rig.step_with_reply() == {"j": 203.0}
    assert len(rig.runner.current_chunk) == 5


def test_the_drop_is_bounded_by_what_the_old_chunk_could_play(rig_factory):
    gate = threading.Event()
    rig = rig_factory(_FakeAdapter(gates={2: gate}))
    rig.start()
    # The old chunk runs dry (5 left at fire) and the robot holds for 6 more ticks.
    played = [rig.step()["j"] for _ in range(14)]
    assert played[:8] == [100.0 + i for i in range(8)]
    assert played[8:] == [107.0] * 6  # late chunk: the last command stands
    gate.set()
    # 12 frames in flight, but only 5 could be consumed: 205, not the chunk's end.
    assert rig.step_with_reply() == {"j": 205.0}


def test_the_drop_keeps_refetch_after_actions(rig_factory):
    gate = threading.Event()
    # Request 2 answers with only 4 actions: at most 4 - refetch_after = 1 may be dropped.
    rig = rig_factory(_FakeAdapter(gates={2: gate}, horizon=lambda n: 4 if n == 2 else HORIZON))
    rig.start()
    for _ in range(5):
        rig.step()
    gate.set()
    assert rig.step_with_reply() == {"j": 201.0}
    assert len(rig.runner.current_chunk) == 3


def test_seam_is_crossfaded_over_blend_frames(rig_factory):
    rig = rig_factory(blend=2)
    rig.start()
    assert [rig.step()["j"] for _ in range(3)] == [100.0, 101.0, 102.0]  # fires request 2
    # One frame in flight: 200 dropped; 201 / 202 blend with the old chunk's 103 / 104.
    first = rig.step_with_reply()
    assert first == {"j": pytest.approx(103.0 * 2 / 3 + 201.0 / 3)}
    assert rig.step() == {"j": pytest.approx(104.0 / 3 + 202.0 * 2 / 3)}
    assert rig.step() == {"j": 203.0}


def test_horizon_is_cut_on_arrival_and_settings_apply_live(rig_factory):
    rig = rig_factory()
    rig.runner.set_target("h1", 5555, "task A", "")
    rig.runner.resume(dict(POSE))
    rig.step()
    rig.wait_reply()
    # The request is already in flight when the horizon shrinks; the cut still applies.
    rig.runner.apply_settings(_settings(seq=1, horizon=4, refetch=2, speed=150))
    assert rig.runner.period_s() == pytest.approx(1 / 45)
    assert rig.step() == {"j": 100.0}
    assert len(rig.runner.current_chunk) == 4
    assert rig.runner.settings.seq == 1


# --------------------------------------------------------------------------- fire


def test_refetch_fires_after_refetch_after_actions(rig_factory):
    rig = rig_factory()
    rig.start()
    rig.step()
    rig.step()
    assert rig.observations == 1  # 2 consumed: not yet
    rig.step()
    assert rig.observations == 2 and rig.runner.inflight


def test_stall_guard_keeps_refetching_a_policy_with_short_chunks(rig_factory):
    # 2-step chunks never reach refetch_after = 3: the guard fires at the chunk's end.
    rig = rig_factory(_FakeAdapter(horizon=2))
    rig.start()
    for _ in range(12):
        if rig.runner.inflight:
            rig.wait_reply()
        rig.step()
    assert len(rig.adapter.calls) >= 4
    assert [v for v in rig.values() if v is not None][:2] == [100.0, 101.0]


def test_late_chunk_keeps_approaching_the_last_command(rig_factory):
    gate = threading.Event()
    rig = rig_factory(_FakeAdapter(gates={2: gate}), smoothing=60)
    rig.start()
    for _ in range(8):
        rig.step()
    command = rig.runner.last_command
    assert command == {"j": 107.0}
    late = [rig.step()["j"] for _ in range(10)]
    gate.set()
    # The smoothed arm lagged behind 107 and keeps closing in on it, never past it.
    assert all(b >= a for a, b in zip(late, late[1:], strict=False))
    assert late[-1] > late[0] and late[-1] <= 107.0


# --------------------------------------------------------------------------- errors / targets


def test_a_rejected_server_is_an_error_and_resume_retries_it(rig_factory, caplog):
    connect = _Connect(error=PolicyConnectError("Cannot reach policy server at h1:5555"))
    rig = rig_factory(connect=connect)
    with caplog.at_level("INFO"):
        rig.runner.set_target("h1", 5555, "task A", "")
        rig.step()
        assert rig.step_with_reply() is None
    assert rig.runner.policy_state == "error"
    assert rig.runner.policy_error == "Cannot reach policy server at h1:5555"
    assert '"seq": 1, "accepted": false' in caplog.text
    # The rejected seq is never asked again.
    for _ in range(5):
        rig.step()
    assert len(connect.calls) == 1 and rig.adapter.calls == []

    connect.error = None
    rig.runner.resume(dict(POSE))  # Run retries under a new seq
    assert rig.runner.policy_state == "connecting"
    assert rig.step() is None
    assert rig.step_with_reply() == {"j": 100.0}
    assert rig.runner.policy_state == "ready"
    assert [t.seq for t in connect.calls] == [1, 2]


def test_a_failed_request_holds_and_reports_the_error(rig_factory):
    rig = rig_factory(_FakeAdapter(fail_on=(2,)))
    rig.start()
    assert [rig.step()["j"] for _ in range(3)] == [100.0, 101.0, 102.0]  # fires request 2, which fails
    assert rig.step_with_reply() is None  # the rest of chunk 1 is dropped: hold
    assert rig.runner.policy_state == "error"
    assert "policy request failed" in (rig.runner.policy_error or "")
    for _ in range(5):
        assert rig.step() is None
    assert len(rig.adapter.calls) == 2  # nothing fires on hold

    rig.runner.resume(dict(POSE))
    rig.step()
    assert rig.step_with_reply() == {"j": 300.0}
    # The server may be gone: the retry probes it again even though the endpoint is the same.
    assert len(rig.connect.calls) == 2


def test_an_instruction_only_change_keeps_the_connection(rig_factory):
    rig = rig_factory()
    rig.start()
    rig.step()  # 100
    rig.runner.set_target("h1", 5555, "task B", "t1")
    assert rig.runner.policy_state == "connecting"
    rig.step()  # 101; fires at once under the new target
    assert rig.runner.inflight
    rig.step_with_reply()
    assert rig.runner.policy_state == "ready"
    assert rig.adapter.calls == ["task A", "task B"]
    assert len(rig.connect.calls) == 1  # same endpoint: no reconnect
    assert rig.connect.clients[0].closed == 0


def test_another_endpoint_reconnects_and_closes_the_old_client(rig_factory):
    rig = rig_factory()
    rig.start()
    rig.step()
    rig.runner.set_target("h2", 5555, "task A", "t1")
    rig.step()
    rig.step_with_reply()
    assert [t.policy_host for t in rig.connect.calls] == ["h1", "h2"]
    assert rig.connect.clients[0].closed == 1 and rig.connect.clients[1].closed == 0


# --------------------------------------------------------------------------- suspend / resume


def test_suspend_stops_at_once_and_drops_the_reply_in_flight(rig_factory):
    gate = threading.Event()
    rig = rig_factory(_FakeAdapter(gates={2: gate}))
    rig.start()
    for _ in range(3):
        rig.step()  # request 2 is in flight
    rig.runner.suspend()
    assert not rig.runner.active
    assert rig.step() is None  # the rest of chunk 1 is gone
    gate.set()
    assert rig.step_with_reply() is None  # chunk 2 answered a pose the robot has left
    assert rig.runner.current_chunk == []
    for _ in range(3):
        assert rig.step() is None
    assert rig.observations == 2  # nothing was requested after the suspend


def test_resume_while_a_reply_is_in_flight_waits_for_a_fresh_chunk(rig_factory):
    gate = threading.Event()
    rig = rig_factory(_FakeAdapter(gates={2: gate}))
    rig.start()
    for _ in range(3):
        rig.step()
    rig.runner.suspend()
    rig.runner.resume({"j": 50.0})  # e.g. after a rewind
    assert rig.step() is None  # the line is busy: nothing fires yet
    assert rig.observations == 2
    gate.set()
    assert rig.step_with_reply() is None  # the stale chunk is dropped, request 3 fires
    assert rig.observations == 3 and rig.runner.inflight
    assert rig.step_with_reply() == {"j": 300.0}


def test_resume_reseeds_the_smoothing_at_the_hold_pose(rig_factory):
    rig = rig_factory(smoothing=80)
    rig.runner.set_target("h1", 5555, "task A", "")
    rig.runner.resume({"j": 90.0})
    rig.step()
    first = rig.step_with_reply()
    # From 90 toward the chunk's 100, not a jump onto it.
    assert first is not None and 90.0 < first["j"] < 100.0


def test_close_stops_the_worker_and_closes_the_client():
    rig = _Rig()
    rig.start()
    rig.step()
    rig.close()
    assert not rig.runner._worker.is_alive()
    assert rig.connect.clients[0].closed == 1
    assert rig.adapter.policy is None
