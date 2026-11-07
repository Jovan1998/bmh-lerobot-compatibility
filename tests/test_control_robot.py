#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_calibrate import CalibrateConfig, calibrate
from lerobot.scripts.lerobot_record import RecordConfig, record, record_loop
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.cycle_timer import CycleTimer
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig

pytestmark = [pytest.mark.dataset, pytest.mark.hardware]


def _ticks(summary: str) -> int:
    """Sample size out of a cadence report — every other number is an average over it."""
    return int(re.search(r"(\d+) ticks", summary).group(1))


def _step_calls(summary: str, step: str) -> int:
    """How many ticks ran *step*, off the loop-body breakdown of a run summary."""
    return int(re.search(rf"\n\s+{step}\s+.*· (\d+) calls", summary).group(1))


def test_calibrate():
    robot_cfg = MockRobotConfig()
    cfg = CalibrateConfig(robot=robot_cfg)
    calibrate(cfg)


def test_teleoperate(cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    cfg = TeleoperateConfig(
        robot=robot_cfg,
        teleop=teleop_cfg,
        fps=30,
        teleop_time_s=0.1,
    )
    teleoperate(cfg)

    # A teleop session has no episodes, so there is one cadence block for the whole run,
    # and the steps it names are the ones the loop wraps.
    (summary,) = cadence_log
    assert summary.startswith("Cadence summary — whole run · target 30 Hz (33.3 ms budget per tick):")
    for step in ("observe", "teleop", "send"):
        assert step in summary, step


def test_record_and_resume(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record",
        num_episodes=1,
        episode_time_s=0.1,
        reset_time_s=0,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    dataset = record(cfg)

    assert dataset.fps == 30
    assert dataset.meta.total_episodes == dataset.num_episodes == 1
    assert dataset.meta.total_frames == dataset.num_frames == 3
    assert dataset.meta.total_tasks == 1

    cfg.resume = True
    # Mock the revision to prevent Hub calls during resume
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record")
        dataset = record(cfg)

    assert dataset.meta.total_episodes == dataset.num_episodes == 2
    assert dataset.meta.total_frames == dataset.num_frames == 6
    assert dataset.meta.total_tasks == 1


def test_record_and_replay(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    record_dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "record_and_replay",
        num_episodes=1,
        episode_time_s=0.1,
        push_to_hub=False,
    )
    record_cfg = RecordConfig(
        robot=robot_cfg,
        dataset=record_dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )
    replay_dataset_cfg = DatasetReplayConfig(
        repo_id=DUMMY_REPO_ID,
        episode=0,
        root=tmp_path / "record_and_replay",
    )
    replay_cfg = ReplayConfig(
        robot=robot_cfg,
        dataset=replay_dataset_cfg,
        play_sounds=False,
    )

    record(record_cfg)

    # Mock the revision to prevent Hub calls during replay
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.dataset_metadata.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "record_and_replay")
        replay(replay_cfg)

    # Replay has to hit the dataset's frame rate or the trajectory plays back at the
    # wrong speed, so it reports its cadence like every other loop.  Its block is the
    # last one and names its own steps.
    assert cadence_log[-1].startswith("Cadence summary — whole run · target 30 Hz")
    assert "read_frame" in cadence_log[-1]


def test_record_reports_a_cadence_summary_per_episode_and_for_the_run(tmp_path, cadence_log):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "cadence",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        play_sounds=False,
    )

    record(cfg)

    assert len(cadence_log) == 3
    per_episode, run = cadence_log[:2], cadence_log[2]
    assert [m.split(":")[0] for m in per_episode] == ["Cadence (episode 0)", "Cadence (episode 1)"]
    assert run.startswith("Cadence summary — whole run, 2 episodes")
    # Windows partition the session, so the episodes account for every tick of the run...
    assert _ticks(run) == sum(_ticks(m) for m in per_episode)
    # ...and every one of those ticks wrote a frame.  The reset phase paces at the same
    # fps but records nothing, so it runs on its own timer rather than diluting the
    # numbers that answer "did I record at `fps`?".
    assert _step_calls(run, "record") == _step_calls(run, "observe") == _ticks(run)


def test_record_forwards_compressed_images_setting_to_reset_phase(tmp_path):
    robot_cfg = MockRobotConfig()
    teleop_cfg = MockTeleopConfig()
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "compressed_images",
        num_episodes=2,
        episode_time_s=0.1,
        reset_time_s=0.1,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=robot_cfg,
        dataset=dataset_cfg,
        teleop=teleop_cfg,
        display_compressed_images=True,
        play_sounds=False,
    )

    with patch("lerobot.scripts.lerobot_record.record_loop", wraps=record_loop) as mock_record_loop:
        record(cfg)

    # Recording episode 0, resetting, then recording episode 1 should all use the same
    # image representation so visualization backends do not receive mixed message types.
    assert [
        call.kwargs.get("display_compressed_images", False) for call in mock_record_loop.call_args_list
    ] == [True, True, True]


def test_record_loop_without_a_teleoperator_paces_and_terminates():
    # Regression: the no-teleop branch used to `continue` past both the pacing sleep and
    # the `timestamp` update, so a reset phase with no teleop device spun as fast as the
    # CPU allowed and never reached `control_time_s` at all.
    robot = make_robot_from_config(MockRobotConfig())
    robot.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    calls = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal calls
        calls += 1
        assert calls <= 20, "loop is spinning: 20 iterations of a 0.1 s phase at 30 Hz"
        return real_get_observation()

    robot.get_observation = counted_get_observation

    try:
        record_loop(
            robot=robot,
            events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=None,
            control_time_s=0.1,
        )
    finally:
        robot.disconnect()

    # 0.1 s at 30 Hz is 3 ticks; the upper bound is what proves the phase was paced.
    assert 1 <= calls <= 6


# --- BMH: `control_fps` — command the robot faster than frames are recorded ---


def _events() -> dict[str, bool]:
    return {"exit_early": False, "stop_recording": False, "rerecord_episode": False}


def _record_config(control_fps: int | None) -> RecordConfig:
    return RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID, single_task="Dummy task", fps=30, push_to_hub=False
        ),
        teleop=MockTeleopConfig(),
        control_fps=control_fps,
        play_sounds=False,
    )


@pytest.mark.parametrize("control_fps", [45, 15, 0, -30])
def test_record_config_rejects_a_control_fps_that_is_not_a_positive_multiple_of_dataset_fps(control_fps):
    with pytest.raises(
        ValueError, match=rf"control_fps \({control_fps}\) must be a positive multiple of dataset.fps \(30\)"
    ):
        _record_config(control_fps)


@pytest.mark.parametrize(("control_fps", "multiplier"), [(None, 1), (30, 1), (60, 2), (90, 3)])
def test_record_config_derives_the_control_multiplier(control_fps, multiplier):
    assert _record_config(control_fps).control_multiplier == multiplier


def test_record_with_control_fps_commands_every_tick_and_records_every_other(tmp_path, cadence_log):
    dataset_cfg = DatasetRecordConfig(
        repo_id=DUMMY_REPO_ID,
        single_task="Dummy task",
        root=tmp_path / "control_fps",
        fps=30,
        num_episodes=1,
        episode_time_s=0.3,
        push_to_hub=False,
    )
    cfg = RecordConfig(
        robot=MockRobotConfig(),
        dataset=dataset_cfg,
        teleop=MockTeleopConfig(),
        control_fps=60,
        play_sounds=False,
    )

    dataset = record(cfg)

    # One episode, so no reset phase: the run summary covers exactly the recorded ticks.
    run = cadence_log[-1]
    assert run.startswith("Cadence summary — whole run")
    assert "Hz commands" in run
    ticks = _ticks(run)
    # Every tick commands the robot; every second one, starting with the first, records.
    assert _step_calls(run, "send") == ticks
    assert _step_calls(run, "record") == math.ceil(ticks / 2)
    # The robot is observed only on the ticks that record.
    assert _step_calls(run, "observe") == _step_calls(run, "record")
    assert dataset.fps == 30
    assert dataset.num_frames == _step_calls(run, "record")


class _TickSpy(CycleTimer):
    """Records the `new_cycle` flag of every tick, and ends the loop (`exit_early`) once
    `stop_at` ticks have run in total, so each call runs an exact number of ticks."""

    def __init__(self, events: dict[str, bool], multiplier: int):
        super().__init__(30, multiplier, records_data=False)
        self.events = events
        self.new_cycles: list[bool] = []
        self.stop_at = 0

    def tick(self, new_cycle: bool = False) -> None:
        self.new_cycles.append(new_cycle)
        super().tick(new_cycle=new_cycle)

    def wait(self) -> None:
        super().wait()
        if len(self.new_cycles) >= self.stop_at:
            self.events["exit_early"] = True


def test_record_loop_starts_a_new_cycle_on_every_record_tick():
    robot = make_robot_from_config(MockRobotConfig())
    teleop = make_teleoperator_from_config(MockTeleopConfig())
    robot.connect()
    teleop.connect()
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    observations = 0
    real_get_observation = robot.get_observation

    def counted_get_observation():
        nonlocal observations
        observations += 1
        return real_get_observation()

    robot.get_observation = counted_get_observation
    events = _events()
    spy = _TickSpy(events, multiplier=2)

    def run(ticks: int) -> None:
        spy.stop_at = len(spy.new_cycles) + ticks
        record_loop(
            robot=robot,
            events=events,
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=1.0,  # safety cap; `stop_at` ends each call first
            timer=spy,
            control_multiplier=2,
        )

    try:
        # Like `record()` across episodes: one timer, and the first call ends mid-group.
        run(3)
        run(4)
    finally:
        robot.disconnect()
        teleop.disconnect()

    # Each call starts on a record tick that re-anchors the pacing, so recorded frames
    # stay 1/fps apart even after an episode that ended after an odd number of ticks.
    assert spy.new_cycles == [True, False, True] + [True, False, True, False]
    assert observations == 2 + 2


def test_record_loop_rejects_a_timer_with_a_different_multiplier():
    robot = make_robot_from_config(MockRobotConfig())
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    with pytest.raises(ValueError, match="control_multiplier is 2"):
        record_loop(
            robot=robot,
            events=_events(),
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            control_time_s=0.1,
            timer=CycleTimer(30),
            control_multiplier=2,
        )


def test_record_loop_rejects_a_control_multiplier_below_one():
    robot = make_robot_from_config(MockRobotConfig())
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    with pytest.raises(ValueError, match="control_multiplier must be >= 1"):
        record_loop(
            robot=robot,
            events=_events(),
            fps=30,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            control_time_s=0.1,
            control_multiplier=0,
        )
