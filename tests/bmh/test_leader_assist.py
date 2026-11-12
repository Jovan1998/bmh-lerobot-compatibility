"""DAgger leader assist (`bmh/dagger/leader_assist.py`): the rules the leader host enforces.

The bus is a fake that records every call and checks the safety property itself: no
joint's limit is raised above the assist's while its torque is on, and no joint comes on
before its limit is written. Time is passed in by hand.
"""

import json

import pytest

from bmh.dagger.leader_assist import (
    DEFAULT_MAX_TORQUE_LIMIT,
    AssistGoal,
    AssistRelease,
    LeaderAssist,
    parse_message,
)

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper"]
JOINTS = ARM  # what the host passes: every motor but the head's
MOTORS = ARM + ["head_pan", "head_tilt"]
PRESENT = {f"{m}.pos": float(i) for i, m in enumerate(MOTORS)}


class _Bus:
    def __init__(self):
        self.motors = dict.fromkeys(MOTORS)
        self.limits = dict.fromkeys(MOTORS, 1000) | {"gripper": 500}
        self.torque_on: set[str] = set()
        self.goal: dict[str, float] = {}
        self.calls: list[tuple] = []
        self.fail_on: str | None = None

    def _maybe_fail(self, what: str) -> None:
        if self.fail_on == what:
            raise ConnectionError(f"no status packet ({what})")

    def read(self, data_name, motor, *, normalize=True):
        assert data_name == "Torque_Limit" and normalize is False
        self._maybe_fail("read")
        self.calls.append(("read", motor))
        return self.limits[motor]

    def write(self, data_name, motor, value, *, normalize=True):
        self._maybe_fail(data_name)
        if data_name == "Torque_Limit":
            assert normalize is False
            if motor in self.torque_on and value > 200:
                raise AssertionError(f"limit of {motor} raised to {value} while its torque is on")
            self.limits[motor] = value
        else:
            assert data_name == "Goal_Position" and normalize is True
            self.goal[motor] = value
        self.calls.append(("write", data_name, motor, value))

    def enable_torque(self, motor):
        self._maybe_fail("enable")
        assert self.limits[motor] <= 200, f"{motor} enabled at Torque_Limit {self.limits[motor]}"
        assert motor in self.goal, f"{motor} enabled without a pinned goal"
        self.torque_on.add(motor)
        self.calls.append(("on", motor))

    def disable_torque(self, motors):
        self._maybe_fail("disable")
        self.torque_on.difference_update(motors)
        self.calls.append(("off", tuple(motors)))

    def sync_write(self, data_name, values):
        assert data_name == "Goal_Position"
        self._maybe_fail("sync_write")
        self.goal.update(values)
        self.calls.append(("goals", dict(values)))


def _goal(value: float, limit: int = 100, joints=JOINTS) -> AssistGoal:
    return AssistGoal(goal=dict.fromkeys(joints, value), torque_limit=limit)


def _engaged(bus: _Bus, assist: LeaderAssist, t0: float = 10.0, limit: int = 100) -> float:
    """Engage and tick until every joint is on; returns the time of the last tick."""
    t = t0
    assist.update(t, PRESENT, _goal(PRESENT["shoulder_pan.pos"], limit))
    while len(bus.torque_on) < len(JOINTS):
        t += 0.05
        assist.update(t, PRESENT, _goal(PRESENT["shoulder_pan.pos"], limit))
    return t


# --------------------------------------------------------------------------- parse_message


def test_parse_goal_keeps_only_the_assisted_joints():
    raw = json.dumps(
        {"goal": {"shoulder_pan.pos": 12, "gripper.pos": 30.5, "head_pan.pos": 5.0}, "torque_limit": 150}
    )
    assert parse_message(raw, JOINTS) == AssistGoal({"shoulder_pan": 12.0, "gripper": 30.5}, 150)


def test_parse_release_and_goal_without_assisted_joints():
    assert parse_message('{"release": true}', JOINTS) == AssistRelease()
    # An arm whose group is held gets a goal with nothing for it: it lets go.
    assert parse_message('{"goal": {"head_pan.pos": 1.0}, "torque_limit": 100}', JOINTS) == AssistRelease()


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1, 2]",
        '{"goal": {"shoulder_pan.pos": 1.0}}',
        '{"goal": {"shoulder_pan.pos": 1.0}, "torque_limit": 1.5}',
        '{"goal": {"shoulder_pan.pos": 1.0}, "torque_limit": true}',
        '{"goal": {"shoulder_pan.pos": "x"}, "torque_limit": 100}',
        '{"goal": {"shoulder_pan.pos": NaN}, "torque_limit": 100}',
        '{"goal": [], "torque_limit": 100}',
    ],
)
def test_parse_rejects_a_broken_message(raw):
    assert parse_message(raw, JOINTS) is None


# --------------------------------------------------------------------------- engaging


def test_limits_first_then_one_joint_per_stagger_each_pinned_where_it_is():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS, 200)
    assist.update(10.0, PRESENT, _goal(50.0, 120))
    assert assist.engaged
    first_on = bus.calls.index(("on", "shoulder_pan"))
    before = bus.calls[:first_on]
    assert before[: len(JOINTS)] == [("read", m) for m in JOINTS]
    assert before[len(JOINTS) : 2 * len(JOINTS)] == [("write", "Torque_Limit", m, 120) for m in JOINTS]
    assert before[-1] == ("write", "Goal_Position", "shoulder_pan", PRESENT["shoulder_pan.pos"])
    assert bus.torque_on == {"shoulder_pan"}

    assist.update(10.03, PRESENT, _goal(50.0, 120))
    assert bus.torque_on == {"shoulder_pan"}  # 30 ms < 50 ms stagger
    assist.update(10.05, PRESENT, _goal(50.0, 120))
    assert bus.torque_on == {"shoulder_pan", "shoulder_lift"}
    t = 10.05
    while len(bus.torque_on) < len(JOINTS):
        t += 0.05
        assist.update(t, PRESENT, _goal(50.0, 120))
    assert [c[1] for c in bus.calls if c[0] == "on"] == JOINTS
    assert "head_pan" not in bus.torque_on and "head_tilt" not in bus.torque_on


def test_no_joint_is_enabled_on_a_tick_without_positions():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS)
    assist.update(10.0, None, _goal(5.0))
    assert assist.engaged and bus.torque_on == set()
    assist.update(10.1, PRESENT, _goal(5.0))
    assert bus.torque_on == {"shoulder_pan"}


def test_limit_is_clamped_to_the_cap():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS, 150)
    assist.update(10.0, PRESENT, _goal(0.0, 900))
    assert {bus.limits[m] for m in JOINTS} == {150}
    assist.update(10.05, PRESENT, _goal(0.0, 0))
    assert {bus.limits[m] for m in JOINTS} == {1}  # never 0 or less


def test_a_new_limit_while_engaged_is_written_to_every_joint():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS, 200)
    t = _engaged(bus, assist, limit=100)
    assist.update(t + 0.02, PRESENT, _goal(PRESENT["shoulder_pan.pos"], 180))
    assert {bus.limits[m] for m in JOINTS} == {180}
    assert bus.torque_on == set(JOINTS)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"joints": []}, "at least one joint"),
        ({"joints": ["shoulder_pan", "nope"]}, "no such motors"),
        ({"joints": JOINTS, "max_torque_limit": 0}, "max_torque_limit"),
        ({"joints": JOINTS, "max_torque_limit": 1001}, "max_torque_limit"),
    ],
)
def test_constructor_rejects_bad_arguments(kwargs, error):
    with pytest.raises(ValueError, match=error):
        LeaderAssist(_Bus(), **kwargs)


def test_default_cap_is_the_bench_tested_200():
    assert DEFAULT_MAX_TORQUE_LIMIT == 200


# --------------------------------------------------------------------------- tracking


def test_goal_walks_toward_the_target_at_max_speed():
    bus = _Bus()
    assist = LeaderAssist(bus, ["shoulder_pan"], max_speed=60.0)
    assist.update(10.0, PRESENT, _goal(30.0, joints=["shoulder_pan"]))
    assert bus.goal["shoulder_pan"] == 0.0  # pinned at present (0.0), dt 0
    goals = []
    t = 10.0
    for _ in range(40):
        t += 1 / 60
        assist.update(t, PRESENT, _goal(30.0, joints=["shoulder_pan"]))
        goals.append(bus.goal["shoulder_pan"])
    steps = [b - a for a, b in zip([0.0, *goals], goals, strict=False)]
    assert max(steps) == pytest.approx(1.0)  # 60 deg/s at 60 Hz
    assert goals[-1] == pytest.approx(30.0)
    assert goals[28] == pytest.approx(29.0)


def test_a_joint_moved_ahead_of_the_goal_is_never_pulled_back():
    bus = _Bus()
    assist = LeaderAssist(bus, ["shoulder_pan"], max_speed=60.0)
    assist.update(10.0, PRESENT, _goal(90.0, joints=["shoulder_pan"]))  # pinned at 0.0
    # The operator moves the joint at 180 deg/s, three times the goal's speed.
    t = 10.0
    for i in range(1, 16):
        t += 1 / 60
        position = 3.0 * i
        assist.update(t, {"shoulder_pan.pos": position}, _goal(90.0, joints=["shoulder_pan"]))
        assert bus.goal["shoulder_pan"] == pytest.approx(position + 1.0)  # one step ahead, never behind


def test_a_joint_pushed_past_the_target_is_pulled_back_to_it_not_beyond():
    bus = _Bus()
    assist = LeaderAssist(bus, ["shoulder_pan"], max_speed=60.0)
    assist.update(10.0, PRESENT, _goal(10.0, joints=["shoulder_pan"]))  # pinned at 0.0
    assist.update(10.0 + 1 / 60, {"shoulder_pan.pos": 16.0}, _goal(10.0, joints=["shoulder_pan"]))
    assert bus.goal["shoulder_pan"] == pytest.approx(15.0)


def test_a_joint_held_back_lets_the_goal_walk_on():
    bus = _Bus()
    assist = LeaderAssist(bus, ["shoulder_pan"], max_speed=60.0)
    assist.update(10.0, PRESENT, _goal(30.0, joints=["shoulder_pan"]))  # pinned at 0.0
    # The joint lags (low torque, or the operator resists): the goal does not wait for it.
    for i in range(1, 11):
        position = 0.5 * (i - 1)
        assist.update(10.0 + i / 60, {"shoulder_pan.pos": position}, _goal(30.0, joints=["shoulder_pan"]))
    assert bus.goal["shoulder_pan"] == pytest.approx(10.0)


def test_a_stalled_loop_does_not_jump_the_goal():
    bus = _Bus()
    assist = LeaderAssist(bus, ["shoulder_pan"], max_speed=60.0, timeout_s=5.0)
    assist.update(10.0, PRESENT, _goal(90.0, joints=["shoulder_pan"]))
    assist.update(12.0, PRESENT, _goal(90.0, joints=["shoulder_pan"]))
    assert bus.goal["shoulder_pan"] == pytest.approx(6.0)  # 0.1 s worth, not 2 s


def test_joint_goals_only_for_enabled_joints():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS)
    assist.update(10.0, PRESENT, _goal(50.0))
    assist.update(10.01, PRESENT, _goal(50.0))
    goal_writes = [c[1] for c in bus.calls if c[0] == "goals"]
    assert goal_writes and all(set(w) == {"shoulder_pan"} for w in goal_writes)


# --------------------------------------------------------------------------- releasing


def test_release_message_turns_torque_off_then_restores_the_limits_read():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS)
    t = _engaged(bus, assist)
    bus.calls.clear()
    assist.update(t + 0.01, PRESENT, AssistRelease())
    assert not assist.engaged and bus.torque_on == set()
    assert bus.calls[0] == ("off", tuple(JOINTS))
    assert bus.calls[1:] == [("write", "Torque_Limit", m, 500 if m == "gripper" else 1000) for m in JOINTS]


def test_watchdog_releases_after_the_messages_stop():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS, timeout_s=0.3)
    t = _engaged(bus, assist)
    assist.update(t + 0.29, PRESENT, None)
    assert assist.engaged
    assist.update(t + 0.31, PRESENT, None)
    assert not assist.engaged and bus.torque_on == set()
    # The next goal engages again, with the limits read the first time kept for restoring.
    assist.update(t + 1.0, PRESENT, _goal(1.0))
    assert assist.engaged
    assert [c for c in bus.calls if c[0] == "read"] == [("read", m) for m in JOINTS]


def test_a_bus_error_releases_and_goals_are_ignored_until_a_release():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS)
    t = _engaged(bus, assist)
    bus.fail_on = "sync_write"
    assist.update(t + 0.02, PRESENT, _goal(3.0))  # never raises
    assert not assist.engaged and bus.torque_on == set()
    bus.fail_on = None
    assist.update(t + 0.04, PRESENT, _goal(3.0))
    assert not assist.engaged  # faulted: not retried at the host's rate
    assist.update(t + 0.06, PRESENT, AssistRelease())
    assist.update(t + 0.08, PRESENT, _goal(3.0))
    assert assist.engaged


def test_a_pause_in_the_messages_also_clears_a_fault():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS, timeout_s=0.3)
    bus.fail_on = "enable"
    assist.update(10.0, PRESENT, _goal(3.0))
    assert not assist.engaged
    bus.fail_on = None
    assist.update(10.5, PRESENT, None)  # the follower stopped sending
    assist.update(11.0, PRESENT, _goal(3.0))
    assert assist.engaged and bus.torque_on == {"shoulder_pan"}


def test_limits_stay_low_when_the_torque_cannot_be_turned_off():
    bus = _Bus()
    assist = LeaderAssist(bus, JOINTS)
    t = _engaged(bus, assist)
    bus.fail_on = "disable"
    assist.update(t + 0.01, PRESENT, AssistRelease())
    assert not assist.engaged
    assert {bus.limits[m] for m in JOINTS} == {100}  # a joint still on must not get full torque


def test_a_failed_engage_releases_what_it_wrote():
    bus = _Bus()
    bus.fail_on = "enable"
    assist = LeaderAssist(bus, JOINTS)
    assist.update(10.0, PRESENT, _goal(3.0))
    assert not assist.engaged
    assert ("off", tuple(JOINTS)) in bus.calls
    assert bus.limits["gripper"] == 500 and bus.limits["shoulder_pan"] == 1000


def test_release_without_engage_touches_nothing():
    bus = _Bus()
    LeaderAssist(bus, JOINTS).release()
    assert bus.calls == []
