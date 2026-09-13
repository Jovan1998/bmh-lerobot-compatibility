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

    # BMH-101 group locks (left arm / right arm / head). Path to a JSON file
    # {"left": bool, "right": bool, "head": bool} written by the controller-app.
    # Polled with one os.stat per get_action(); only re-read when it changed.
    # None disables locking entirely (upstream behaviour).
    lock_file: str | None = None
    # Seconds to ease a group from its held pose back to the live leader pose after unlock.
    unlock_blend_s: float = 0.8
