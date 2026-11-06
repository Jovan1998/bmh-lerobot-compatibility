"""BMH: Unix signals stand in for the recording keys (headless Pi, controller-app UI).

The controller-app ends an episode, re-records or stops a running `bmh-record` / `lerobot-record`
by sending SIGUSR1 / SIGUSR2 / SIGTERM to its PID. `record()` installs those handlers right after
`init_keyboard_listener()`, whatever keyboard backend that picked (or none at all), and puts the
previous handlers back when it returns (see `test_record_guards.py`).
"""

import os
import signal
import time
from unittest.mock import patch

import pytest

from lerobot.utils import keyboard_input

_SIGNALS = (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM)


@pytest.fixture
def restore_signal_handlers():
    previous = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    yield
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def _init_keyboard() -> dict:
    # Never start a real keyboard listener from a test (pynput would hook the desktop).
    with patch.object(keyboard_input, "create_key_listener", return_value=None):
        listener, events = keyboard_input.init_keyboard_listener()
    assert listener is None
    return events


@pytest.fixture
def events(restore_signal_handlers):
    """`events` as `record()` sets them up: keyboard listener (none here), then the signals."""
    events = _init_keyboard()
    keyboard_input.install_recording_signal_handlers(events)
    return events


def _send(sig: int) -> None:
    os.kill(os.getpid(), sig)
    # Python runs the handler on the main thread at the next check for pending signals,
    # which `time.sleep` performs.
    time.sleep(0.01)


def test_events_start_cleared(events):
    assert events == {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


def test_sigusr1_ends_episode(events):
    _send(signal.SIGUSR1)
    assert events == {"exit_early": True, "rerecord_episode": False, "stop_recording": False}


def test_sigusr2_rerecords(events):
    _send(signal.SIGUSR2)
    assert events == {"exit_early": True, "rerecord_episode": True, "stop_recording": False}


def test_sigterm_stops_recording_gracefully(events):
    _send(signal.SIGTERM)
    # The process is still alive (we are running) — SIGTERM did not kill it.
    assert events == {"exit_early": True, "rerecord_episode": False, "stop_recording": True}


def test_handlers_installed_for_all_three_signals(events):
    for sig in _SIGNALS:
        handler = signal.getsignal(sig)
        assert callable(handler), f"{sig!r} has no Python handler"
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)


def test_log_tells_the_operator_how_to_signal(restore_signal_handlers, caplog):
    with caplog.at_level("INFO"):
        keyboard_input.install_recording_signal_handlers(_init_keyboard())
    pid = str(os.getpid())
    assert f"kill -USR1 {pid}" in caplog.text
    assert f"kill -USR2 {pid}" in caplog.text
    assert f"kill -TERM {pid}" in caplog.text


def test_restore_puts_the_previous_handlers_back(restore_signal_handlers):
    sentinel = lambda signum, frame: None  # noqa: E731
    for sig in _SIGNALS:
        signal.signal(sig, sentinel)

    restore = keyboard_input.install_recording_signal_handlers(_init_keyboard())
    assert all(signal.getsignal(sig) is not sentinel for sig in _SIGNALS)
    restore()

    assert all(signal.getsignal(sig) is sentinel for sig in _SIGNALS)


def test_keyboard_listener_alone_leaves_signals_alone(restore_signal_handlers):
    """Other callers of `init_keyboard_listener` (`lerobot-rollout`) own SIGTERM themselves."""
    sentinel = lambda signum, frame: None  # noqa: E731
    for sig in _SIGNALS:
        signal.signal(sig, sentinel)

    _init_keyboard()

    assert all(signal.getsignal(sig) is sentinel for sig in _SIGNALS)
