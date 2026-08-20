import json
import logging
from functools import cached_property

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


class SONetworkLeader(Teleoperator):
    """
    Receives SO leader arm positions over ZMQ instead of reading from local USB.
    Drop-in replacement for SOLeader on the follower Pi.

    Pair with so_leader_host.py running on the leader Pi.
    """

    config_class = SONetworkLeaderConfig
    name = "so_network_leader"

    def __init__(self, config: SONetworkLeaderConfig):
        super().__init__(config)
        self.config = config

        self.zmq_context = None
        self.zmq_socket = None
        self._is_connected = False
        self._last_action: dict[str, float] | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        motor_names = ARM_MOTOR_NAMES + (HEAD_MOTOR_NAMES if self.config.with_head else [])
        return {f"{motor}.pos": float for motor in motor_names}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

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
            try:
                self._last_action = json.loads(last_msg)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode leader message: {e}")

        return self._last_action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        self._is_connected = False
        logger.info(f"{self} disconnected.")
