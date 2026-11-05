"""BMH: Unix signals stand in for the recording keys (headless Pi, controller-app UI).

The controller-app ends an episode, re-records or stops a running `bmh-record` / `lerobot-record`
by sending SIGUSR1 / SIGUSR2 / SIGTERM to its PID. `init_keyboard_listener()` must install those
handlers whatever keyboard backend it picks (or none at all).
"""

import importlib
import os
import signal
import time
from contextlib import ExitStack
from unittest.mock import patch

import pytest

_SIGNALS = (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM)


def _keyboard_module():
    """The module that owns `init_keyboard_listener` (it moved upstream in 0.6)."""
    try:
        return importlib.import_module("lerobot.utils.keyboard_input")
    except ImportError:
        return importlib.import_module("lerobot.common.control_utils")


@pytest.fixture
def events():
    """`events` from `init_keyboard_listener()` with no keyboard backend; signal handlers restored after."""
    module = _keyboard_module()
    previous = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    with ExitStack() as stack:
        # Never start a real keyboard listener from a test (pynput would hook the desktop).
        if hasattr(module, "create_key_listener"):
            stack.enter_context(patch.object(module, "create_key_listener", return_value=None))
        else:
            stack.enter_context(patch.object(module, "is_headless", return_value=True))
        listener, events = module.init_keyboard_listener()
        assert listener is None
        try:
            yield events
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


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


def test_headless_log_names_the_pid(caplog):
    """With no keyboard, the log tells the operator how to drive recording via `kill`."""
    module = _keyboard_module()
    previous = {sig: signal.getsignal(sig) for sig in _SIGNALS}
    with ExitStack() as stack:
        if hasattr(module, "create_key_listener"):
            stack.enter_context(patch.object(module, "create_key_listener", return_value=None))
        else:
            stack.enter_context(patch.object(module, "is_headless", return_value=True))
        try:
            with caplog.at_level("INFO"):
                module.init_keyboard_listener()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    pid = str(os.getpid())
    assert f"kill -USR1 {pid}" in caplog.text
    assert f"kill -USR2 {pid}" in caplog.text
    assert f"kill -TERM {pid}" in caplog.text
