import logging
from functools import cached_property

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_network_leader import SONetworkLeader
from ..so_network_leader.config_so_network_leader import SONetworkLeaderConfig
from .config_bi_so_network_leader import BiSONetworkLeaderConfig

logger = logging.getLogger(__name__)


class BiSONetworkLeader(Teleoperator):
    """
    Bimanual SO network leader: composes two SONetworkLeader instances
    (left arm on one ZMQ port, right arm on another).

    Drop-in replacement for BiSOLeader on the follower Pi.
    """

    config_class = BiSONetworkLeaderConfig
    name = "bi_so_network_leader"

    def __init__(self, config: BiSONetworkLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SONetworkLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            remote_ip=config.left_arm_config.remote_ip,
            port_zmq=config.left_arm_config.port_zmq,
            polling_timeout_ms=config.left_arm_config.polling_timeout_ms,
            connect_timeout_s=config.left_arm_config.connect_timeout_s,
            use_degrees=config.left_arm_config.use_degrees,
            with_head=config.left_arm_config.with_head,
        )

        right_arm_config = SONetworkLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            remote_ip=config.right_arm_config.remote_ip,
            port_zmq=config.right_arm_config.port_zmq,
            polling_timeout_ms=config.right_arm_config.polling_timeout_ms,
            connect_timeout_s=config.right_arm_config.connect_timeout_s,
            use_degrees=config.right_arm_config.use_degrees,
            with_head=config.right_arm_config.with_head,
        )

        self.left_arm = SONetworkLeader(left_arm_config)
        self.right_arm = SONetworkLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_features = self.left_arm.action_features
        right_features = self.right_arm.action_features
        return {
            **{f"left_{k}": v for k, v in left_features.items()},
            **{f"right_{k}": v for k, v in right_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        action_dict = {}

        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
