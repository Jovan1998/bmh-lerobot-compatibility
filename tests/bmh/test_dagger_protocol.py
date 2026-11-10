"""`bmh-dagger` wire contract: stdin command lines in, `BMH_DAGGER` / ack lines out."""

import io
import json
import time

import pytest

from bmh.dagger.protocol import (
    ACK_PREFIX,
    STATUS_PREFIX,
    Align,
    Cancel,
    Discard,
    Eof,
    Finish,
    Home,
    Invalid,
    Pause,
    Policy,
    Record,
    Run,
    Save,
    StdinReader,
    Step,
    format_ack,
    format_align,
    format_status,
    parse_command,
    parse_line,
)


def _line(**fields) -> str:
    return json.dumps(fields)


# --------------------------------------------------------------------------- accept


@pytest.mark.parametrize(
    "cmd, cls",
    [
        ("run", Run),
        ("pause", Pause),
        ("record", Record),
        ("cancel", Cancel),
        ("save", Save),
        ("discard", Discard),
        ("finish", Finish),
    ],
)
def test_plain_commands(cmd, cls):
    assert parse_command(_line(id=7, cmd=cmd)) == cls(7)


def test_policy_is_stripped_and_hides_the_token():
    cmd = parse_command(
        _line(
            id=1,
            cmd="policy",
            policy_host=" 1.2.3.4 ",
            policy_port=5555,
            api_token=" secret ",
            lang_instruction="  put the cube in the bowl ",
        )
    )
    assert cmd == Policy(1, "1.2.3.4", 5555, "put the cube in the bowl", "secret")
    assert "secret" not in repr(cmd)


def test_policy_token_is_optional():
    cmd = parse_command(_line(id=1, cmd="policy", policy_host="h", policy_port=1, lang_instruction="x"))
    assert isinstance(cmd, Policy) and cmd.api_token == ""


def test_step_home_and_align():
    assert parse_command(_line(id=4, cmd="step", cursor=45)) == Step(4, 45)
    assert parse_command(_line(id=4, cmd="step", cursor=0)) == Step(4, 0)
    assert parse_command(_line(id=5, cmd="home", pose={"left_shoulder_pan.pos": 0, "g": 2.5})) == Home(
        5, {"left_shoulder_pan.pos": 0.0, "g": 2.5}
    )
    assert parse_command(_line(id=6, cmd="align", task=" fold ")) == Align(6, "fold", True, False, 10)
    assert parse_command(
        _line(id=6, cmd="align", task="fold", auto_start=False, assist=True, assist_torque_pct=30)
    ) == Align(6, "fold", False, True, 30)


# --------------------------------------------------------------------------- reject


@pytest.mark.parametrize(
    "line, match",
    [
        ("not json", "not JSON"),
        ("[1]", "expected a JSON object"),
        (_line(cmd="run"), "'id' must be a non-negative integer"),
        (_line(id="1", cmd="run"), "'id' must be"),
        (_line(id=True, cmd="run"), "'id' must be"),
        (_line(id=-1, cmd="run"), "'id' must be"),
        (_line(id=1), "unknown cmd None"),
        (_line(id=1, cmd="dance"), "unknown cmd 'dance'"),
        (_line(id=1, cmd="step"), "'cursor' must be a non-negative integer"),
        (_line(id=1, cmd="step", cursor=-1), "'cursor' must be"),
        (_line(id=1, cmd="step", cursor=1.5), "'cursor' must be"),
        (_line(id=1, cmd="step", cursor=True), "'cursor' must be"),
        (_line(id=1, cmd="home"), "'pose' must be a non-empty object"),
        (_line(id=1, cmd="home", pose={}), "'pose' must be a non-empty object"),
        (_line(id=1, cmd="home", pose={"j": "1"}), "must be a finite number"),
        (_line(id=1, cmd="home", pose={"j": True}), "must be a finite number"),
        ('{"id": 1, "cmd": "home", "pose": {"j": NaN}}', "must be a finite number"),
        (_line(id=1, cmd="align"), "'task' must be a non-empty string"),
        (_line(id=1, cmd="align", task="  "), "'task' must be a non-empty string"),
        (_line(id=1, cmd="align", task="t", assist_torque_pct=0), "'assist_torque_pct' must be"),
        (_line(id=1, cmd="align", task="t", assist_torque_pct=31), "'assist_torque_pct' must be"),
        (_line(id=1, cmd="align", task="t", assist_torque_pct=10.0), "'assist_torque_pct' must be"),
        (_line(id=1, cmd="align", task="t", auto_start="yes"), "'auto_start' must be a boolean"),
        (_line(id=1, cmd="align", task="t", assist=1), "'assist' must be a boolean"),
        (_line(id=1, cmd="policy", policy_port=1, lang_instruction="x"), "'policy_host' must be"),
        (_line(id=1, cmd="policy", policy_host="h", lang_instruction="x"), "'policy_port' must be"),
        (
            _line(id=1, cmd="policy", policy_host="h", policy_port=70000, lang_instruction="x"),
            "'policy_port' must be",
        ),
        (_line(id=1, cmd="policy", policy_host="h", policy_port=1), "'lang_instruction' must be"),
        (
            _line(id=1, cmd="policy", policy_host="h", policy_port=1, lang_instruction="x", api_token=5),
            "'api_token' must be a string",
        ),
    ],
)
def test_rejections_name_the_problem(line, match):
    with pytest.raises(ValueError, match=match):
        parse_command(line)


def test_parse_line_keeps_the_id_of_a_rejected_command():
    assert parse_line(_line(id=9, cmd="dance")) == Invalid("unknown cmd 'dance'", 9)
    assert parse_line("garbage").id is None
    assert parse_line(_line(cmd="run")).id is None
    assert parse_line(_line(id=3, cmd="run")) == Run(3)


# --------------------------------------------------------------------------- output


def _payload(line: str, prefix: str) -> dict:
    assert line.startswith(prefix)
    return json.loads(line[len(prefix) :])


def test_ack_lines():
    assert format_ack(6, True) == 'BMH_DAGGER_ACK {"id": 6, "ok": true}'
    assert format_ack(7, False, "not aligning") == (
        'BMH_DAGGER_ACK {"id": 7, "ok": false, "error": "not aligning"}'
    )
    assert _payload(format_ack(1, True), ACK_PREFIX) == {"id": 1, "ok": True}


def test_status_line_minimal():
    line = format_status("running", "ready", 412, 0, 3)
    assert (
        line
        == 'BMH_DAGGER {"phase": "running", "policy": "ready", "history": 412, "cursor": 0, "episodes": 3}'
    )


def test_status_line_with_hold_align_and_take_in_fixed_order():
    align = format_align({"left_shoulder_pan.pos": -3.2049, "g": 0.004}, False, 14.4, "off")
    line = format_status(
        "aligning",
        "ready",
        412,
        0,
        3,
        hold={"left_shoulder_pan.pos": 12.3456},
        align=align,
        take_frames=0,
    )
    payload = _payload(line, STATUS_PREFIX)
    assert payload == {
        "phase": "aligning",
        "policy": "ready",
        "history": 412,
        "cursor": 0,
        "episodes": 3,
        "hold": {"left_shoulder_pan.pos": 12.35},
        "align": {
            "diff": {"left_shoulder_pan.pos": -3.2, "g": 0.0},
            "aligned": False,
            "leader_age_ms": 14,
            "assist": "off",
        },
        "take": {"frames": 0},
    }
    assert list(payload) == ["phase", "policy", "history", "cursor", "episodes", "hold", "align", "take"]
    assert list(payload["align"]) == ["diff", "aligned", "leader_age_ms", "assist"]
    assert "\n" not in line


# --------------------------------------------------------------------------- stdin


def _drain_until(reader: StdinReader, count: int) -> list:
    items: list = []
    deadline = time.monotonic() + 5
    while len(items) < count:
        assert time.monotonic() < deadline, f"only {items} arrived"
        items.extend(reader.drain())
        time.sleep(0.005)
    return items


def test_stdin_reader_delivers_lines_in_order_then_eof():
    stream = io.StringIO(
        "\n".join([_line(id=1, cmd="run"), "", "garbage", _line(id=2, cmd="step", cursor=3)]) + "\n"
    )
    reader = StdinReader(stream)
    assert reader.drain() == []
    reader.start()
    items = _drain_until(reader, 4)
    assert items[0] == Run(1)
    assert isinstance(items[1], Invalid) and items[1].id is None
    assert items[2] == Step(2, 3)
    assert items[3] == Eof()
    assert reader.drain() == []
