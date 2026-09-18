"""Unit tests for the BMH-101 inference control file - no ZMQ, no hardware.

Covers ``parse_target`` validation, ``InferenceTarget.endpoint`` equality,
``ControlFileWatcher`` change detection (stat-based, like the teleop lock watcher)
and the ``BMH_TARGET`` status line. Pure stdlib module, so this runs anywhere.
Run with ``uv run --no-sync pytest tests/bmh/test_inference_control.py -q``.
"""

import json
import os

import pytest

from bmh.groot_client.control import (
    STATUS_PREFIX,
    ControlFileWatcher,
    InferenceTarget,
    format_target_status,
    parse_target,
)

VALID = {
    "seq": 1,
    "policy_host": "10.0.0.2",
    "policy_port": 5555,
    "lang_instruction": "pick up the cube",
    "api_token": "s3cret",
}


def _write(path, data, mtime_ns: int) -> None:
    path.write_text(json.dumps(data))
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _without(key: str) -> dict:
    return {k: v for k, v in VALID.items() if k != key}


# --------------------------------------------------------------------------- parse


def test_parse_target_accepts_valid_and_strips_whitespace():
    target = parse_target(
        {**VALID, "policy_host": " 10.0.0.2 ", "lang_instruction": " pick ", "api_token": " tok "}
    )
    assert target == InferenceTarget(1, "10.0.0.2", 5555, "pick", "tok")


def test_parse_target_token_defaults_to_empty():
    assert parse_target(_without("api_token")).api_token == ""


@pytest.mark.parametrize(
    "bad",
    [
        "not a dict",
        _without("seq"),
        {**VALID, "seq": True},
        {**VALID, "seq": "1"},
        {**VALID, "seq": -1},
        _without("policy_host"),
        {**VALID, "policy_host": ""},
        {**VALID, "policy_host": "   "},
        {**VALID, "policy_host": 5},
        _without("policy_port"),
        {**VALID, "policy_port": 0},
        {**VALID, "policy_port": 65536},
        {**VALID, "policy_port": "5555"},
        {**VALID, "policy_port": True},
        _without("lang_instruction"),
        {**VALID, "lang_instruction": ""},
        {**VALID, "lang_instruction": None},
        {**VALID, "api_token": 7},
    ],
)
def test_parse_target_rejects(bad):
    with pytest.raises(ValueError):
        parse_target(bad)


# --------------------------------------------------------------------------- target


def test_endpoint_ignores_prompt_and_seq():
    base = InferenceTarget(1, "h", 5555, "task A", "tok")
    assert InferenceTarget(2, "h", 5555, "task B", "tok").endpoint == base.endpoint
    assert InferenceTarget(2, "h2", 5555, "task A", "tok").endpoint != base.endpoint
    assert InferenceTarget(2, "h", 5556, "task A", "tok").endpoint != base.endpoint
    assert InferenceTarget(2, "h", 5555, "task A", "other").endpoint != base.endpoint


def test_repr_hides_token():
    assert "s3cret" not in repr(InferenceTarget(1, "h", 5555, "task", "s3cret"))


# --------------------------------------------------------------------------- watcher


def test_watcher_missing_file_reports_none(tmp_path):
    watcher = ControlFileWatcher(tmp_path / "control.json")
    assert watcher.read_initial() is None
    assert watcher.poll() is None
    assert watcher.poll() is None


def test_read_initial_primes_change_detection(tmp_path):
    path = tmp_path / "control.json"
    _write(path, VALID, 1_000)
    watcher = ControlFileWatcher(path)
    assert watcher.read_initial() == parse_target(VALID)
    assert watcher.poll() is None  # the startup read is not a change
    _write(path, {**VALID, "seq": 2}, 2_000)
    assert watcher.poll().seq == 2
    assert watcher.poll() is None


def test_read_initial_invalid_file_raises(tmp_path):
    path = tmp_path / "control.json"
    path.write_text("{not json")
    with pytest.raises(ValueError):
        ControlFileWatcher(path).read_initial()
    path.write_text(json.dumps({**VALID, "policy_port": 0}))
    with pytest.raises(ValueError, match="policy_port"):
        ControlFileWatcher(path).read_initial()


def test_poll_reads_only_when_file_changes(tmp_path):
    path = tmp_path / "control.json"
    watcher = ControlFileWatcher(path)
    _write(path, VALID, 1_000)
    assert watcher.poll() == parse_target(VALID)
    assert watcher.poll() is None
    _write(path, VALID, 1_000)  # same signature -> not a change
    assert watcher.poll() is None
    _write(path, {**VALID, "seq": 2, "lang_instruction": "next"}, 2_000)
    assert watcher.poll() == InferenceTarget(2, "10.0.0.2", 5555, "next", "s3cret")
    assert watcher.poll() is None


def test_poll_malformed_file_warns_once_and_keeps_none(tmp_path, caplog):
    path = tmp_path / "control.json"
    watcher = ControlFileWatcher(path)
    _write(path, VALID, 1_000)
    assert watcher.poll() is not None

    path.write_text("{not json")
    os.utime(path, ns=(2_000, 2_000))
    with caplog.at_level("WARNING"):
        assert watcher.poll() is None
        assert watcher.poll() is None
    assert caplog.text.lower().count("control file") == 1

    _write(path, {**VALID, "seq": 3}, 3_000)
    assert watcher.poll().seq == 3


def test_poll_deleted_then_recreated(tmp_path):
    path = tmp_path / "control.json"
    watcher = ControlFileWatcher(path)
    _write(path, VALID, 1_000)
    assert watcher.poll() is not None
    path.unlink()
    assert watcher.poll() is None
    assert watcher.poll() is None
    _write(path, {**VALID, "seq": 2}, 3_000)
    assert watcher.poll().seq == 2


# --------------------------------------------------------------------------- status


def test_format_target_status_round_trips_without_token():
    target = InferenceTarget(3, "h", 5555, 'say "hi" ✓', "s3cret")

    rejected = format_target_status(target, False, "Cannot reach")
    assert rejected.startswith(STATUS_PREFIX)
    assert "\n" not in rejected
    assert "s3cret" not in rejected
    assert json.loads(rejected[len(STATUS_PREFIX) :]) == {
        "seq": 3,
        "accepted": False,
        "error": "Cannot reach",
        "policy_host": "h",
        "policy_port": 5555,
        "lang_instruction": 'say "hi" ✓',
    }

    accepted = json.loads(format_target_status(target, True, None)[len(STATUS_PREFIX) :])
    assert accepted["accepted"] is True
    assert accepted["error"] is None
