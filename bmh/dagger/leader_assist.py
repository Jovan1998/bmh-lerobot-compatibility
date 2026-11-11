"""Leader assist: a leader arm pulls lightly toward the pose the follower holds.

While the operator lines the leader arms up with the robot for a DAgger take-over,
`bmh-dagger` on the follower Pi sends, every tick, the pose the robot holds and a torque
limit to the leader host (`bmh/scripts/so_leader_host.py`), one ZMQ socket per arm:

.. code-block:: json

    {"goal": {"shoulder_pan.pos": 12.0, "gripper.pos": 31.5}, "torque_limit": 100}
    {"release": true}

`LeaderAssist` runs inside the leader host, one per arm, and is where the safety rules are
enforced, not by the sender:

- Only the joints it was built with are ever driven (the host passes the arm joints with the
  gripper, never the head); goal keys of other joints are ignored.
- The limit is clamped to ``max_torque_limit`` and written to every joint before any torque
  comes on. 1000 is the servo's full torque.
- The joints come on one at a time, ``stagger_s`` apart, each with its goal pinned to where it
  is, so no joint jumps and the inrush currents do not add up on the leader's small supply.
  This happens across ticks: the position stream never stalls.
- The goal walks toward the target at ``max_speed`` per second (degrees; percent for the
  gripper), so a far target is approached slowly instead of with a jump.
- Without a message for ``timeout_s`` (the follower process died, the network dropped) the
  torque goes off by itself: the leader can never stay stiff.
- A bus error releases, and goals are ignored until a ``release`` message or a pause in the
  messages, so a failing bus is not retried at the host's rate.

On release the torque goes off first and the limits read before the first engage are written
back after; a joint whose torque could not be turned off keeps the low limit.

stdlib only; the bus is any object with the `SerialMotorsBus` calls used here.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Full torque in the servos' `Torque_Limit` register.
TORQUE_LIMIT_MAX = 1000
# The leader host's cap unless told otherwise: what the leader arms were bench-tested with
# (2026-11-12, both arms at once, the leader Pi stayed up and the arms could be overpowered).
DEFAULT_MAX_TORQUE_LIMIT = 200
TIMEOUT_S = 0.3
MAX_SPEED = 60.0
STAGGER_S = 0.05
# A longer gap between two ticks is treated as this long, so a stalled loop never turns
# into one big step of the goal.
_MAX_TICK_S = 0.1


@dataclass(frozen=True)
class AssistGoal:
    """Pull toward ``goal`` (motor name -> position) with ``torque_limit`` (0..1000)."""

    goal: dict[str, float]
    torque_limit: int


@dataclass(frozen=True)
class AssistRelease:
    """Torque off."""


def parse_message(raw: str | bytes, joints: Iterable[str]) -> AssistGoal | AssistRelease | None:
    """Decode one feedback message for an arm whose assisted motors are ``joints``.

    Goal keys are action keys (``shoulder_pan.pos``); keys of other motors are dropped. A goal
    that names none of ``joints`` releases the arm (the follower sends that for an arm whose
    group is held, so it must not be pulled).

    Returns:
        The message, or ``None`` (logged) when it breaks the format.
    """
    try:
        data = json.loads(raw)
    except ValueError as e:
        logger.warning("Ignoring a leader-assist message that is not JSON: %s", e)
        return None
    if not isinstance(data, dict):
        logger.warning("Ignoring a leader-assist message that is not an object: %r", data)
        return None
    if data.get("release") is True:
        return AssistRelease()
    goal, limit = data.get("goal"), data.get("torque_limit")
    if not isinstance(goal, dict) or type(limit) is not int:
        logger.warning("Ignoring a leader-assist message without goal / torque_limit: %r", data)
        return None
    wanted = set(joints)
    positions: dict[str, float] = {}
    for key, value in goal.items():
        motor = str(key).removesuffix(".pos")
        if motor not in wanted:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            logger.warning("Ignoring a leader-assist message with a bad position for %s: %r", key, value)
            return None
        positions[motor] = float(value)
    if not positions:
        return AssistRelease()
    return AssistGoal(goal=positions, torque_limit=limit)


class LeaderAssist:
    """Drives some joints of one leader arm weakly toward a goal; see the module docstring.

    Call `update` once per leader-host tick, with the positions just read and the latest
    message (or ``None``), and `release` on shutdown. Neither raises.
    """

    def __init__(
        self,
        bus: Any,
        joints: Iterable[str],
        max_torque_limit: int = DEFAULT_MAX_TORQUE_LIMIT,
        *,
        timeout_s: float = TIMEOUT_S,
        max_speed: float = MAX_SPEED,
        stagger_s: float = STAGGER_S,
        name: str = "leader",
    ):
        """Bind to a connected bus; nothing is written before the first goal.

        Args:
            bus: The arm's motors bus.
            joints: Motor names the assist may drive.
            max_torque_limit: Highest ``Torque_Limit`` written, 1..1000.
            timeout_s: Release after this long without a message.
            max_speed: Speed of the goal toward the target, per second.
            stagger_s: Pause between two joints' torque coming on.
            name: For the log.
        """
        self._joints = list(joints)
        if not self._joints:
            raise ValueError("LeaderAssist needs at least one joint")
        unknown = [m for m in self._joints if m not in bus.motors]
        if unknown:
            raise ValueError(f"LeaderAssist: no such motors on the bus: {unknown}")
        if not 1 <= max_torque_limit <= TORQUE_LIMIT_MAX:
            raise ValueError(f"max_torque_limit must be in [1, {TORQUE_LIMIT_MAX}], got {max_torque_limit}")
        self._bus = bus
        self._max_torque_limit = max_torque_limit
        self._timeout_s = timeout_s
        self._max_speed = max_speed
        self._stagger_s = stagger_s
        self._name = name

        self._engaged = False
        self._faulted = False
        self._restore: dict[str, int] | None = None
        self._limit = 0
        self._pending: list[str] = []
        self._enabled: list[str] = []
        self._goal: dict[str, float] = {}
        self._target: dict[str, float] = {}
        self._last_message_at: float | None = None
        self._last_enable_at = -math.inf
        self._last_track_at = 0.0

    @property
    def joints(self) -> list[str]:
        """The motors the assist may drive."""
        return list(self._joints)

    @property
    def engaged(self) -> bool:
        """Whether the limits are written and the joints are (being) torqued."""
        return self._engaged

    def update(
        self,
        now: float,
        present: Mapping[str, float] | None,
        message: AssistGoal | AssistRelease | None,
    ) -> None:
        """One tick: take the message, run the watchdog, enable the next joint, move the goal.

        Args:
            now: Monotonic seconds.
            present: The arm's positions read this tick (action keys, ``shoulder_pan.pos``),
                ``None`` when the read failed (no joint is enabled then).
            message: The latest message since the previous tick, if any.
        """
        try:
            self._update(now, present, message)
        except Exception:
            logger.exception("%s: leader assist failed — releasing", self._name)
            self._faulted = True
            self.release()

    def release(self) -> None:
        """Torque off, then the previous limits back. Best effort, never raises."""
        if not self._engaged:
            return
        self._engaged = False
        torque_off = True
        try:
            self._bus.disable_torque(self._joints)
        except Exception:
            torque_off = False
            logger.warning(
                "%s: could not turn the assist's torque off; keeping the low Torque_Limit",
                self._name,
                exc_info=True,
            )
        if torque_off:
            for motor, value in (self._restore or {}).items():
                try:
                    self._bus.write("Torque_Limit", motor, value, normalize=False)
                except Exception:
                    logger.warning("%s: failed to restore Torque_Limit=%d on %s", self._name, value, motor)
        self._pending = []
        self._enabled = []
        self._goal = {}
        self._target = {}
        logger.info("%s: leader assist off", self._name)

    def _update(
        self,
        now: float,
        present: Mapping[str, float] | None,
        message: AssistGoal | AssistRelease | None,
    ) -> None:
        if isinstance(message, AssistRelease):
            self._last_message_at = now
            self._faulted = False
            self.release()
            return
        if isinstance(message, AssistGoal):
            self._last_message_at = now
            if self._faulted:
                return
            limit = min(max(message.torque_limit, 1), self._max_torque_limit)
            if not self._engaged:
                self._engage(limit, now)
            elif limit != self._limit:
                self._write_limit(limit)
            self._target = dict(message.goal)
        elif self._last_message_at is not None and now - self._last_message_at > self._timeout_s:
            self._last_message_at = None
            self._faulted = False
            if self._engaged:
                logger.warning("%s: no assist message for %.1f s — releasing", self._name, self._timeout_s)
                self.release()
            return
        if not self._engaged:
            return
        self._enable_next(now, present)
        self._track(now)

    def _engage(self, limit: int, now: float) -> None:
        if self._restore is None:
            self._restore = {
                motor: int(self._bus.read("Torque_Limit", motor, normalize=False)) for motor in self._joints
            }
        self._engaged = True
        # The limit first: no joint may ever see full torque, not even for one packet.
        self._write_limit(limit)
        self._pending = list(self._joints)
        self._enabled = []
        self._goal = {}
        self._last_enable_at = -math.inf
        self._last_track_at = now
        logger.info(
            "%s: leader assist on (Torque_Limit %d, joints %s)", self._name, limit, ",".join(self._joints)
        )

    def _write_limit(self, limit: int) -> None:
        for motor in self._joints:
            self._bus.write("Torque_Limit", motor, limit, normalize=False)
        self._limit = limit

    def _enable_next(self, now: float, present: Mapping[str, float] | None) -> None:
        if not self._pending or present is None or now - self._last_enable_at < self._stagger_s:
            return
        motor = self._pending[0]
        position = present.get(f"{motor}.pos")
        if position is None:
            return
        # Pinned to where the joint is, so it does not move when its torque comes on.
        self._bus.write("Goal_Position", motor, position)
        self._bus.enable_torque(motor)
        self._pending.pop(0)
        self._enabled.append(motor)
        self._goal[motor] = float(position)
        self._last_enable_at = now

    def _track(self, now: float) -> None:
        dt = min(max(now - self._last_track_at, 0.0), _MAX_TICK_S)
        self._last_track_at = now
        if not self._enabled:
            return
        step = self._max_speed * dt
        for motor in self._enabled:
            target = self._target.get(motor)
            if target is not None:
                current = self._goal[motor]
                self._goal[motor] = current + min(max(target - current, -step), step)
        self._bus.sync_write("Goal_Position", {motor: self._goal[motor] for motor in self._enabled})
