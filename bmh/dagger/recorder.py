"""The staging dataset of a DAgger session, built the way `lerobot-record` builds it.

`open_dataset` makes the same calls as ``record()`` in
``src/lerobot/scripts/lerobot_record.py`` (default processors, the two feature
aggregations, ``stamp_repo_id``, ``LeRobotDataset.create`` with the same arguments), so a
DAgger staging dataset has exactly the features of a recording staging dataset and the
controller-app's converter merges it unchanged. `TakeRecorder` writes one correction as
one episode, frame by frame like ``record_loop``.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots import Robot
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

logger = logging.getLogger(__name__)


def open_dataset(cfg: DatasetRecordConfig, robot: Robot) -> LeRobotDataset:
    """Create the session's dataset at ``cfg.root`` (a fresh folder), as ``record()`` does.

    Raises:
        ValueError: For an ``eval_`` repo name (reserved for policy evaluation, as in
            ``lerobot-record``).
    """
    teleop_action_processor, _robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.video,
        ),
    )
    repo_name = cfg.repo_id.split("/", 1)[-1]
    if repo_name.startswith("eval_"):
        raise ValueError("Dataset names starting with 'eval_' are reserved for policy evaluation.")
    cfg.stamp_repo_id()
    num_cameras = len(robot.cameras)
    return LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=cfg.root,
        robot_type=robot.robot_type,
        features=dataset_features,
        use_videos=cfg.video,
        image_writer_processes=cfg.num_image_writer_processes,
        image_writer_threads=cfg.num_image_writer_threads_per_camera * num_cameras,
        batch_encoding_size=cfg.video_encoding_batch_size,
        rgb_encoder=cfg.rgb_encoder,
        depth_encoder=cfg.depth_encoder,
        encoder_threads=cfg.encoder_threads,
        streaming_encoding=cfg.streaming_encoding,
        encoder_queue_maxsize=cfg.encoder_queue_maxsize,
    )


class TakeRecorder:
    """Writes the frames of the running take; `save` makes them an episode, `discard` drops them."""

    def __init__(self, dataset: LeRobotDataset):
        """Write into ``dataset`` (opened with `open_dataset`)."""
        self._dataset = dataset
        self.frames = 0

    @property
    def episodes(self) -> int:
        """Episodes saved so far."""
        return self._dataset.num_episodes

    def add(self, observation: dict[str, Any], action: dict[str, float], task: str) -> None:
        """Append one frame: the robot's observation and the action that was sent."""
        observation_frame = build_dataset_frame(self._dataset.features, observation, prefix=OBS_STR)
        action_frame = build_dataset_frame(self._dataset.features, action, prefix=ACTION)
        self._dataset.add_frame({**observation_frame, **action_frame, "task": task})
        self.frames += 1

    def save(self) -> bool:
        """Save the take as one episode (blocks while videos encode).

        Returns:
            ``False`` when the take has no frame: nothing is written.
        """
        self.frames = 0
        if not self._dataset.has_pending_frames():
            logger.warning("No frames were recorded for this take, skipping save.")
            return False
        self._dataset.save_episode()
        return True

    def discard(self) -> None:
        """Drop the take's frames (and their temporary images)."""
        self.frames = 0
        if self._dataset.has_pending_frames():
            self._dataset.clear_episode_buffer()
