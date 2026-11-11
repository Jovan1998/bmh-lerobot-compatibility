import logging
import time
from functools import cached_property
from pathlib import Path

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_network_leader import SONetworkLeader
from ..so_network_leader.config_so_network_leader import SONetworkLeaderConfig
from ..so_network_leader.so_network_leader import DEFAULT_FEEDBACK_TORQUE_LIMIT
from .config_bi_so_network_leader import BiSONetworkLeaderConfig
from .group_lock import ActionGroupLock, LockFileWatcher, bimanual_lock_groups, head_jitter_offsets

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
            feedback_port_zmq=config.left_arm_config.feedback_port_zmq,
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
            feedback_port_zmq=config.right_arm_config.feedback_port_zmq,
        )

        self.left_arm = SONetworkLeader(left_arm_config)
        self.right_arm = SONetworkLeader(right_arm_config)

        # BMH-101 group locks (left arm / right arm / head), driven by a JSON state file
        # the controller-app writes. Disabled (pure passthrough) when lock_file is None.
        self._lock_groups = bimanual_lock_groups(left_with_head=config.left_arm_config.with_head)
        self._lock = ActionGroupLock(
            self._lock_groups,
            blend_s=config.unlock_blend_s,
            pose_blend_s=config.home_blend_s,
        )
        self._lock_watcher = (
            LockFileWatcher(Path(config.lock_file).expanduser()) if config.lock_file else None
        )
        if self._lock_watcher is not None:
            logger.info(
                f"Teleop group locks enabled via {self._lock_watcher.path} "
                f"(unlock blend {config.unlock_blend_s:.2f}s, home blend {config.home_blend_s:.2f}s)"
            )

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_features = self.left_arm.action_features
        right_features = self.right_arm.action_features
        return {
            **{f"left_{k}": v for k, v in left_features.items()},
            **{f"right_{k}": v for k, v in right_features.items()},
        }

    @property
    def has_feedback(self) -> bool:
        """BMH: whether both arms have a feedback channel to the leader host."""
        return self.left_arm.has_feedback and self.right_arm.has_feedback

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        if not self.has_feedback:
            return {}
        return {
            **{f"left_{k}": v for k, v in self.left_arm.feedback_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.feedback_features.items()},
        }

    @property
    def locked_keys(self) -> frozenset[str]:
        """BMH: action keys of the groups that are locked right now (they do not follow the leader)."""
        if self._lock_watcher is None:
            return frozenset()
        locks = self._lock.locks
        return frozenset(key for group, keys in self._lock_groups.items() if locks.get(group) for key in keys)

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def last_message_age_s(self) -> float:
        """Seconds since the staler of the two arms last heard from the leader host."""
        return max(self.left_arm.last_message_age_s, self.right_arm.last_message_age_s)

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
            # Head jitter: the offset a frozen head is held at (ignored while the head tracks).
            head_jitter = self._lock_watcher.head_jitter
            self._lock.set_offsets(head_jitter_offsets(head_jitter))
            # Home position: held groups listed in the lock pose ease to it instead of freezing.
            self._lock.set_lock_pose(self._lock_watcher.lock_pose)
            logger.info(
                "Teleop locks: %s | head jitter: %s | home: %s",
                " ".join(f"{group}={'on' if locked else 'off'}" for group, locked in new_locks.items()),
                " ".join(f"{axis}={degrees:+.1f}" for axis, degrees in head_jitter.items()),
                ",".join(self._lock.posed_groups()) or "none",
            )
        return self._lock.apply(action_dict, now)

    @check_if_not_connected
    def send_feedback(
        self, feedback: dict[str, float], torque_limit: int = DEFAULT_FEEDBACK_TORQUE_LIMIT
    ) -> None:
        """BMH: pull the leader arms toward ``feedback`` (``left_`` / ``right_`` action keys).

        An arm none of whose keys is in ``feedback`` is released, so leaving an arm out lets
        go of it at once. See `SONetworkLeader.send_feedback`.

        Raises:
            NotImplementedError: Unless both arms have a feedback channel.
        """
        if not self.has_feedback:
            raise NotImplementedError(
                "BiSONetworkLeader has no feedback channel (feedback_port_zmq is not set on both arms)"
            )
        for prefix, arm in (("left_", self.left_arm), ("right_", self.right_arm)):
            part = {
                key.removeprefix(prefix): value for key, value in feedback.items() if key.startswith(prefix)
            }
            if part:
                arm.send_feedback(part, torque_limit)
            else:
                arm.release_feedback()

    def release_feedback(self) -> None:
        """BMH: let go of both leader arms now."""
        self.left_arm.release_feedback()
        self.right_arm.release_feedback()

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
