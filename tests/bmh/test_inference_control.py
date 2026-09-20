"""Unit tests for the BMH-101 inference control file - no ZMQ, no hardware.

Covers ``parse_target`` validation (incl. the idle document), ``InferenceTarget.endpoint``
equality, ``ControlFileWatcher`` change detection (stat-based, like the teleop lock watcher)
and the ``BMH_TARGET`` / ``BMH_FEEDBACK`` status lines. Pure stdlib module, so this runs anywhere.
Run with ``uv run --no-sync pytest tests/bmh/test_inference_control.py -q``.
"""

import json
import os

import pytest

from bmh.groot_client.control import (
    FEEDBACK_STATUS_PREFIX,
    STATUS_PREFIX,
    ControlFileWatcher,
    InferenceTarget,
    format_feedback_status,
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
        {**VALID, "idle": "true"},
        {**VALID, "idle": 1},
        {**VALID, "idle": None},
        {"seq": 1, "idle": "yes"},
        {"idle": True},
        {"seq": -1, "idle": True},
        {"seq": 1, "idle": False},
    ],
)
def test_parse_target_rejects(bad):
    with pytest.raises(ValueError):
        parse_target(bad)


def test_parse_target_idle_needs_only_seq():
    target = parse_target({"seq": 4, "idle": True})
    assert target == InferenceTarget.make_idle(4)
    assert target.idle is True
    assert (target.seq, target.policy_host, target.policy_port, target.lang_instruction) == (4, "", 0, "")
    assert target.api_token == ""


def test_parse_target_idle_ignores_other_keys():
    # Even values that would be rejected on a real target: an idle document is only its seq.
    doc = {**VALID, "seq": 5, "idle": True, "policy_port": "nope", "lang_instruction": 7}
    assert parse_target(doc) == InferenceTarget.make_idle(5)


def test_parse_target_explicit_idle_false_is_a_normal_target():
    target = parse_target({**VALID, "idle": False})
    assert target == parse_target(VALID)
    assert target.idle is False


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


def test_watcher_yields_idle_and_back(tmp_path):
    path = tmp_path / "control.json"
    _write(path, {"seq": 1, "idle": True}, 1_000)
    watcher = ControlFileWatcher(path)
    assert watcher.read_initial() == InferenceTarget.make_idle(1)
    _write(path, {**VALID, "seq": 2}, 2_000)
    assert watcher.poll() == parse_target({**VALID, "seq": 2})
    _write(path, {"seq": 3, "idle": True}, 3_000)
    assert watcher.poll() == InferenceTarget.make_idle(3)
    assert watcher.poll() is None


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
        "idle": False,
        "policy_host": "h",
        "policy_port": 5555,
        "lang_instruction": 'say "hi" ✓',
    }

    accepted = json.loads(format_target_status(target, True, None)[len(STATUS_PREFIX) :])
    assert accepted["accepted"] is True
    assert accepted["error"] is None


def test_format_target_status_idle():
    status = format_target_status(InferenceTarget.make_idle(4), True, None)
    assert json.loads(status[len(STATUS_PREFIX) :]) == {
        "seq": 4,
        "accepted": True,
        "error": None,
        "idle": True,
        "policy_host": "",
        "policy_port": 0,
        "lang_instruction": "",
    }


# --------------------------------------------------------------------------- feedback


def test_format_feedback_status_is_the_exact_compact_line():
    # The controller-app matches /BMH_FEEDBACK (\{.*\})\s*$/ — this is the contract.
    line = format_feedback_status(4, {"done": True}, {"done": 0.93})
    assert line == 'BMH_FEEDBACK {"seq":4,"flags":{"done":true},"values":{"done":0.93}}'


def test_format_feedback_status_round_trips():
    flags = {"done": False, "object_grasped": True}
    line = format_feedback_status(7, flags, {"done": 0.123456, "object_grasped": 1.0})
    assert line.startswith(FEEDBACK_STATUS_PREFIX)
    assert "\n" not in line
    payload = json.loads(line[len(FEEDBACK_STATUS_PREFIX) :])
    assert payload == {"seq": 7, "flags": flags, "values": {"done": 0.123, "object_grasped": 1.0}}
    assert list(payload) == ["seq", "flags", "values"]
    assert list(payload["flags"]) == ["done", "object_grasped"]  # feedbacks[] order survives


def test_format_feedback_status_cleared_snapshot():
    assert format_feedback_status(4, {}, {}) == 'BMH_FEEDBACK {"seq":4,"flags":{},"values":{}}'


def test_format_feedback_status_never_emits_invalid_json():
    line = format_feedback_status(1, {"a": False, "b": False}, {"a": float("nan"), "b": float("inf")})
    assert json.loads(line[len(FEEDBACK_STATUS_PREFIX) :]) == {
        "seq": 1,
        "flags": {"a": False, "b": False},
        "values": {},
    }


def test_feedback_line_is_not_a_target_line():
    assert not format_feedback_status(1, {}, {}).startswith(STATUS_PREFIX)
    assert not format_target_status(InferenceTarget.make_idle(1), True, None).startswith(
        FEEDBACK_STATUS_PREFIX
    )
