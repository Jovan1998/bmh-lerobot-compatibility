from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class SONetworkLeaderBaseConfig:
    """Configuration for receiving SO leader arm positions over ZMQ."""

    # IP address of the leader Pi running so_leader_host
    remote_ip: str = "192.168.0.100"

    # ZMQ port to receive leader positions on
    port_zmq: int = 5555

    # Timeout in ms when polling for new ZMQ messages
    polling_timeout_ms: int = 15

    # Timeout in seconds waiting for initial connection
    connect_timeout_s: int = 5

    # Must match the leader's use_degrees setting
    use_degrees: bool = True

    # BMH-101: head_pan/head_tilt (IDs 8/9) ride the LEFT arm's serial bus.
    # Must match the leader host's with_head setting.
    with_head: bool = False


@TeleoperatorConfig.register_subclass("so_network_leader")
@dataclass
class SONetworkLeaderConfig(TeleoperatorConfig, SONetworkLeaderBaseConfig):
    pass
