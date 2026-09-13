import logging
import time
from functools import cached_property
from pathlib import Path

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_network_leader import SONetworkLeader
from ..so_network_leader.config_so_network_leader import SONetworkLeaderConfig
from .config_bi_so_network_leader import BiSONetworkLeaderConfig
from .group_lock import ActionGroupLock, LockFileWatcher, bimanual_lock_groups

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

        # BMH-101 group locks (left arm / right arm / head), driven by a JSON state file
        # the controller-app writes. Disabled (pure passthrough) when lock_file is None.
        self._lock = ActionGroupLock(
            bimanual_lock_groups(left_with_head=config.left_arm_config.with_head),
            blend_s=config.unlock_blend_s,
        )
        self._lock_watcher = (
            LockFileWatcher(Path(config.lock_file).expanduser()) if config.lock_file else None
        )
        if self._lock_watcher is not None:
            logger.info(
                f"Teleop group locks enabled via {self._lock_watcher.path} "
                f"(unlock blend {config.unlock_blend_s:.2f}s)"
            )

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

        if self._lock_watcher is None:
            return action_dict

        # Pick up toggles from the controller-app's state file (one os.stat per tick; the
        # file is only read when it changed), then hold / blend the locked groups. This runs
        # before lerobot-record stores the action, so datasets contain the pose the follower
        # was actually commanded.
        now = time.monotonic()
        new_locks = self._lock_watcher.poll()
        if new_locks is not None:
            self._lock.set_locks(new_locks)
            logger.info(
                "Teleop locks: %s",
                " ".join(f"{group}={'on' if locked else 'off'}" for group, locked in new_locks.items()),
            )
        return self._lock.apply(action_dict, now)

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
