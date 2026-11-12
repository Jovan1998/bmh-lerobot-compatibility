import contextlib
import json
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from .config_so_network_leader import SONetworkLeaderConfig

logger = logging.getLogger(__name__)

# The motor keys that SOLeader.get_action() produces
ARM_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]
# BMH-101: head_pan/head_tilt ride the LEFT arm's serial bus (with_head=True)
HEAD_MOTOR_NAMES = [
    "head_pan",
    "head_tilt",
]
# BMH: `Torque_Limit` (0..1000) of a feedback goal sent without one: 10 % of full torque.
DEFAULT_FEEDBACK_TORQUE_LIMIT = 100
# BMH: how long closing the feedback socket may wait to deliver the last message (the
# release). The leader host releases by itself 0.3 s after the messages stop anyway.
_FEEDBACK_LINGER_MS = 200


class SONetworkLeader(Teleoperator):
    """
    Receives SO leader arm positions over ZMQ instead of reading from local USB.
    Drop-in replacement for SOLeader on the follower Pi.

    Pair with so_leader_host.py running on the leader Pi.

    BMH: with ``feedback_port_zmq`` set, `send_feedback` pushes goal positions back to the
    leader host, which pulls the leader arm toward them with a low torque limit (the DAgger
    leader assist, see ``bmh/dagger/leader_assist.py``).
    """

    config_class = SONetworkLeaderConfig
    name = "so_network_leader"

    def __init__(self, config: SONetworkLeaderConfig):
        super().__init__(config)
        self.config = config

        # zmq is imported lazily in connect(), so these stay untyped here.
        self.zmq_context: Any = None
        self.zmq_socket: Any = None
        self._feedback_socket: Any = None
        self._is_connected = False
        # Set from the first message in connect(); get_action() is only reachable after that.
        self._last_action: dict[str, float] = {}
        # BMH: `time.perf_counter()` of the last message received (None before connect()).
        self._last_message_at: float | None = None

    @property
    def last_message_age_s(self) -> float:
        """Seconds since the last message from the leader host (``inf`` before the first).

        `get_action()` keeps returning the cached positions when nothing arrives, so a dead
        stream looks like a leader that stands still; this tells the two apart.
        """
        if self._last_message_at is None:
            return float("inf")
        return time.perf_counter() - self._last_message_at

    @cached_property
    def action_features(self) -> dict[str, type]:
        motor_names = ARM_MOTOR_NAMES + (HEAD_MOTOR_NAMES if self.config.with_head else [])
        return {f"{motor}.pos": float for motor in motor_names}

    @property
    def has_feedback(self) -> bool:
        """Whether `send_feedback` can reach the leader host (``feedback_port_zmq`` is set)."""
        return self.config.feedback_port_zmq is not None

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # The leader host never drives the head, so only the arm joints take feedback.
        if not self.has_feedback:
            return {}
        return {f"{motor}.pos": float for motor in ARM_MOTOR_NAMES}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # Calibration lives on leader Pi
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import zmq

        self._zmq = zmq
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_socket.setsockopt(zmq.CONFLATE, 1)

        zmq_url = f"tcp://{self.config.remote_ip}:{self.config.port_zmq}"
        self.zmq_socket.connect(zmq_url)
        logger.info(f"SONetworkLeader connecting to {zmq_url}")

        # Wait for first message to confirm the leader host is alive
        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.config.connect_timeout_s * 1000))
        if self.zmq_socket not in socks or socks[self.zmq_socket] != zmq.POLLIN:
            self.zmq_socket.close()
            self.zmq_context.term()
            raise DeviceNotConnectedError(
                f"Timeout waiting for leader host at {zmq_url} "
                f"(waited {self.config.connect_timeout_s}s). Is so_leader_host running on the leader Pi?"
            )

        # Read and cache the first message
        msg = self.zmq_socket.recv_string(zmq.NOBLOCK)
        self._last_action = json.loads(msg)
        self._last_message_at = time.perf_counter()

        if self.config.feedback_port_zmq is not None:
            self._feedback_socket = self.zmq_context.socket(zmq.PUSH)
            self._feedback_socket.setsockopt(zmq.CONFLATE, 1)
            self._feedback_socket.setsockopt(zmq.LINGER, _FEEDBACK_LINGER_MS)
            feedback_url = f"tcp://{self.config.remote_ip}:{self.config.feedback_port_zmq}"
            self._feedback_socket.connect(feedback_url)
            logger.info(f"SONetworkLeader sends leader-assist feedback to {feedback_url}")
        self._is_connected = True
        logger.info(f"SONetworkLeader connected to {zmq_url}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """Poll ZMQ for latest leader positions. Returns cached action if no new message."""
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.config.polling_timeout_ms))
        except zmq.ZMQError as e:
            logger.error(f"ZMQ polling error: {e}")
            return self._last_action

        if self.zmq_socket not in socks:
            # No new data — return last known positions (arm holds position)
            return self._last_action

        # Drain to latest message (CONFLATE should ensure only 1, but be safe)
        last_msg = None
        while True:
            try:
                last_msg = self.zmq_socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break

        if last_msg is not None:
            self._last_message_at = time.perf_counter()
            try:
                self._last_action = json.loads(last_msg)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode leader message: {e}")

        return self._last_action

    @check_if_not_connected
    def send_feedback(
        self, feedback: dict[str, float], torque_limit: int = DEFAULT_FEEDBACK_TORQUE_LIMIT
    ) -> None:
        """BMH: pull the leader arm toward ``feedback`` (goal positions by action key).

        The leader host drives the arm joints it was asked for with ``torque_limit`` (0..1000,
        capped on its side), ignores the head, and lets go 0.3 s after the last message, so
        call this every tick for as long as the pull should last. Never blocks.

        Raises:
            NotImplementedError: Without ``feedback_port_zmq`` (upstream behaviour).
        """
        if self._feedback_socket is None:
            raise NotImplementedError(
                "SONetworkLeader has no feedback channel (feedback_port_zmq is not set)"
            )
        goal = {key: float(value) for key, value in feedback.items()}
        self._push_feedback({"goal": goal, "torque_limit": int(torque_limit)})

    def release_feedback(self) -> None:
        """BMH: let go of the leader arm now. A no-op without a feedback channel."""
        if self._feedback_socket is not None:
            self._push_feedback({"release": True})

    def _push_feedback(self, message: dict[str, Any]) -> None:
        # zmq.Again: the leader host is not reachable (yet); the next tick tries again.
        with contextlib.suppress(self._zmq.Again):
            self._feedback_socket.send_string(json.dumps(message), self._zmq.NOBLOCK)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._feedback_socket is not None:
            # Never leave the leader pulling; the linger gives the release time to go out.
            self.release_feedback()
            self._feedback_socket.close()
            self._feedback_socket = None
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        self._is_connected = False
        logger.info(f"{self} disconnected.")
