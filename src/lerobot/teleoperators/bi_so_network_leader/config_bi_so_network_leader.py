from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from ..so_network_leader.config_so_network_leader import SONetworkLeaderBaseConfig


@TeleoperatorConfig.register_subclass("bi_so_network_leader")
@dataclass
class BiSONetworkLeaderConfig(TeleoperatorConfig):
    """Configuration for bimanual SO network leader (two arms over ZMQ)."""

    left_arm_config: SONetworkLeaderBaseConfig = field(
        default_factory=lambda: SONetworkLeaderBaseConfig(port_zmq=5555)
    )
    right_arm_config: SONetworkLeaderBaseConfig = field(
        default_factory=lambda: SONetworkLeaderBaseConfig(port_zmq=5557)
    )
