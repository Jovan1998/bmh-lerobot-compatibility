"""Worker-level tests for ``bmh-groot-client`` target switching - no robot, no server.

Drives ``_inference_worker`` with a fake adapter / fake policy clients and checks the
decision rules: a prompt-only change is adopted, an endpoint change is probed once via
``connect`` and replaces the client, a rejected target answers with ``Hold`` and is
never re-probed for the same seq. Also covers ``_connect_policy``'s error mapping and
``PolicyClient.close()`` against a dead port. Needs the ``bmh_network`` extra
(``msgpack-numpy``) because importing the script pulls in ``PolicyClient``; skipped
otherwise. Run with ``uv run --no-sync --with msgpack-numpy pytest tests/bmh -q``.
"""

import queue
import socket
import threading
import time

import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("cv2")
pytest.importorskip("draccus")
zmq = pytest.importorskip("zmq")

import bmh.scripts.bmh_groot_client as script  # noqa: E402
from bmh.groot_client.control import Hold, InferenceTarget  # noqa: E402
from bmh.groot_client.server_client import PolicyClient  # noqa: E402
from bmh.scripts.bmh_groot_client import (  # noqa: E402
    PolicyConnectError,
    _connect_policy,
    _inference_worker,
)

HORIZON = 4
CHUNK_LEN = 8


def _target(seq, host="h1", port=5555, lang="task A", token="t1") -> InferenceTarget:
    return InferenceTarget(seq, host, port, lang, token)


INITIAL = _target(1)


# --------------------------------------------------------------------------- fakes


class _FakeClient:
    """Stands in for PolicyClient: remembers its target and counts close() calls."""

    def __init__(self, target: InferenceTarget):
        self.host = target.policy_host
        self.port = target.policy_port
        self.api_token = target.api_token
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _FakeAdapter:
    """Duck-typed BiSoBimanualAdapter: records which client + instruction served each request."""

    def __init__(self, client):
        self.policy = client
        self.calls: list[tuple[str, int, str, str]] = []

    def get_action(self, obs):
        self.calls.append((self.policy.host, self.policy.port, self.policy.api_token, obs["lang"]))
        return [{"k": float(i)} for i in range(CHUNK_LEN)]


class _Connect:
    """Scripted `connect` callable: one outcome per call (an exception to raise, or None = success)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[InferenceTarget] = []

    def __call__(self, target: InferenceTarget):
        self.calls.append(target)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeClient(target)


class _Worker:
    """Runs `_inference_worker` in a thread; `ask()` sends one request and returns its reply."""

    def __init__(self, connect):
        self.obs_queue: queue.Queue = queue.Queue(maxsize=1)
        self.chunk_queue: queue.Queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.initial_client = _FakeClient(INITIAL)
        self.adapter = _FakeAdapter(self.initial_client)
        self.thread = threading.Thread(
            target=_inference_worker,
            args=(self.adapter, self.obs_queue, self.chunk_queue, HORIZON, self.stop, INITIAL, connect),
            daemon=True,
        )
        self.thread.start()

    def ask(self, target: InferenceTarget):
        self.obs_queue.put((target, {}))
        return self.chunk_queue.get(timeout=5)

    def close(self) -> None:
        self.stop.set()
        self.obs_queue.put(None)
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()


@pytest.fixture
def worker_factory():
    workers: list[_Worker] = []

    def make(connect=None) -> _Worker:
        worker = _Worker(connect or _Connect())
        workers.append(worker)
        return worker

    yield make
    for worker in workers:
        worker.close()


# --------------------------------------------------------------------------- worker


def test_same_seq_serves_active_target_and_truncates(worker_factory):
    worker = worker_factory()
    assert len(worker.ask(INITIAL)) == HORIZON
    worker.ask(INITIAL)
    assert worker.adapter.calls == [("h1", 5555, "t1", "task A")] * 2


def test_prompt_only_change_is_adopted_without_connect(worker_factory):
    connect = _Connect()
    worker = worker_factory(connect)
    assert isinstance(worker.ask(_target(2, lang="task B")), list)
    assert connect.calls == []
    assert worker.adapter.policy is worker.initial_client
    assert worker.adapter.calls[-1] == ("h1", 5555, "t1", "task B")


def test_endpoint_change_connects_once_and_replaces_client(worker_factory):
    connect = _Connect()
    worker = worker_factory(connect)
    new = _target(2, host="h2", token="t2", lang="task B")
    assert isinstance(worker.ask(new), list)
    worker.ask(new)
    assert [t.seq for t in connect.calls] == [2]
    assert worker.adapter.policy is not worker.initial_client
    assert worker.initial_client.closed == 1
    assert worker.adapter.calls[-2:] == [("h2", 5555, "t2", "task B")] * 2


def test_rejected_target_holds_and_is_probed_once(worker_factory, caplog):
    connect = _Connect(PolicyConnectError("Cannot reach h2"))
    worker = worker_factory(connect)
    bad = _target(2, host="h2", lang="task B")
    with caplog.at_level("INFO"):
        replies = [worker.ask(bad) for _ in range(3)]
    assert replies == [Hold(seq=2, error="Cannot reach h2")] * 3
    assert len(connect.calls) == 1
    assert worker.adapter.calls == []  # the rejected prompt never reached a server
    assert worker.adapter.policy is worker.initial_client
    assert worker.initial_client.closed == 0
    assert caplog.text.count("BMH_TARGET") == 1
    assert '"seq": 2, "accepted": false' in caplog.text


def test_unexpected_connect_error_also_holds(worker_factory):
    worker = worker_factory(_Connect(RuntimeError("boom")))
    reply = worker.ask(_target(2, host="h2"))
    assert isinstance(reply, Hold)
    assert reply.seq == 2 and "boom" in reply.error


def test_new_seq_with_same_content_reprobes_and_recovers(worker_factory):
    connect = _Connect(PolicyConnectError("down"), None)
    worker = worker_factory(connect)
    assert isinstance(worker.ask(_target(2, host="h2")), Hold)
    assert isinstance(worker.ask(_target(3, host="h2")), list)
    assert [t.seq for t in connect.calls] == [2, 3]
    assert worker.adapter.calls[-1][:2] == ("h2", 5555)
    assert worker.initial_client.closed == 1


def test_returning_to_still_connected_endpoint_needs_no_connect(worker_factory):
    connect = _Connect(PolicyConnectError("down"))
    worker = worker_factory(connect)
    assert isinstance(worker.ask(_target(2, host="h2")), Hold)
    assert isinstance(worker.ask(_target(3, lang="task C")), list)  # back on h1
    assert len(connect.calls) == 1
    assert worker.adapter.calls[-1] == ("h1", 5555, "t1", "task C")


def test_get_action_failure_signals_none(worker_factory):
    worker = worker_factory()

    def boom(_obs):
        raise RuntimeError("server error")

    worker.adapter.get_action = boom
    assert worker.ask(INITIAL) is None


# --------------------------------------------------------------------------- _connect_policy


class _ProbeClient:
    """PolicyClient stand-in for `_connect_policy`: scripted ping / modality outcomes."""

    instances: list["_ProbeClient"] = []
    ping_result: object = True
    modality: dict = {}

    def __init__(self, host, port, timeout_ms, api_token):
        self.host, self.port, self.api_token = host, port, api_token
        self.timeouts = [timeout_ms]
        self.closed = 0
        _ProbeClient.instances.append(self)

    def ping(self):
        if isinstance(self.ping_result, BaseException):
            raise self.ping_result
        return self.ping_result

    def get_modality_config(self):
        return self.modality

    def set_timeout_ms(self, timeout_ms):
        self.timeouts.append(timeout_ms)

    def close(self):
        self.closed += 1


class _ValidatingAdapter:
    def __init__(self, error: str | None = None):
        self.error = error
        self.seen: list[dict] = []

    def validate_modality(self, cfg):
        self.seen.append(cfg)
        if self.error:
            raise SystemExit(self.error)


@pytest.fixture
def probe_client(monkeypatch):
    monkeypatch.setattr(_ProbeClient, "instances", [])
    monkeypatch.setattr(_ProbeClient, "ping_result", True)
    monkeypatch.setattr(_ProbeClient, "modality", {"state": {"modality_keys": ["x"]}})
    monkeypatch.setattr(script, "PolicyClient", _ProbeClient)
    return _ProbeClient


def test_connect_policy_success_switches_to_request_timeout(probe_client):
    adapter = _ValidatingAdapter()
    client = _connect_policy(_target(1), adapter, probe_timeout_ms=300, timeout_ms=15000)
    assert client is probe_client.instances[0]
    assert client.timeouts == [300, 15000]
    assert client.api_token == "t1"
    assert adapter.seen == [probe_client.modality]
    assert client.closed == 0


def test_connect_policy_passes_no_token_as_none(probe_client):
    client = _connect_policy(_target(1, token=""), _ValidatingAdapter(), 300, 15000)
    assert client.api_token is None


def test_connect_policy_maps_unreachable_and_closes(probe_client):
    probe_client.ping_result = False
    with pytest.raises(PolicyConnectError, match="Cannot reach GR00T policy server at h1:5555"):
        _connect_policy(_target(1), _ValidatingAdapter(), 300, 15000)
    assert probe_client.instances[0].closed == 1


def test_connect_policy_maps_bad_token(probe_client):
    probe_client.ping_result = RuntimeError("Server error: invalid api_token")
    with pytest.raises(PolicyConnectError, match="invalid api_token"):
        _connect_policy(_target(1), _ValidatingAdapter(), 300, 15000)
    assert probe_client.instances[0].closed == 1


def test_connect_policy_converts_modality_system_exit(probe_client):
    with pytest.raises(PolicyConnectError, match="bad layout"):
        _connect_policy(_target(1), _ValidatingAdapter(error="bad layout"), 300, 15000)
    assert probe_client.instances[0].closed == 1


def test_connect_policy_tolerates_missing_modality_endpoint(probe_client, monkeypatch):
    def raising(_self):
        raise RuntimeError("no such endpoint")

    monkeypatch.setattr(_ProbeClient, "get_modality_config", raising)
    adapter = _ValidatingAdapter(error="would fail if called")
    client = _connect_policy(_target(1), adapter, 300, 15000)
    assert adapter.seen == []
    assert client.closed == 0


# --------------------------------------------------------------------------- PolicyClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_policy_client_close_after_failed_ping_is_fast_and_idempotent():
    client = PolicyClient("127.0.0.1", _free_port(), timeout_ms=200)
    first_socket = client.socket
    assert client.ping() is False  # nobody listens: the request times out, socket recreated
    assert first_socket.closed  # the stale REQ socket was closed, not leaked
    tic = time.perf_counter()
    client.close()
    client.close()
    assert time.perf_counter() - tic < 2.0
    assert client.socket.closed


def test_policy_client_set_timeout_ms_applies_to_live_socket():
    client = PolicyClient("127.0.0.1", _free_port(), timeout_ms=200)
    client.set_timeout_ms(50)
    assert client.timeout_ms == 50
    assert client.socket.getsockopt(zmq.RCVTIMEO) == 50
    assert client.socket.getsockopt(zmq.SNDTIMEO) == 50
    client.close()
