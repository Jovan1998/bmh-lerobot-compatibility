"""BMH: guards in `lerobot-record`'s episode loop for signal-driven recording.

The controller-app ends episodes with SIGUSR1 (see `test_recording_signals.py`). Two races:

* a "next episode" signal that lands while no loop runs (e.g. during a long batch video encode
  inside `save_episode`) must not skip the following episode — `exit_early` is cleared at the
  start of every episode;
* a stop/next signal on the very first tick leaves an empty episode buffer — `record()` must
  skip `save_episode()` (which raises on an empty buffer) instead of crashing the session.
"""

import contextlib
import logging
import signal
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets")
pytestmark = pytest.mark.dataset

from lerobot.configs.dataset import DatasetRecordConfig  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: E402
from lerobot.scripts import lerobot_record  # noqa: E402
from lerobot.scripts.lerobot_record import RecordConfig, record  # noqa: E402
from lerobot.teleoperators.so_leader import SO101LeaderConfig  # noqa: E402


class _FakeRobot:
    name = "fake_robot"
    robot_type = "fake_robot"
    action_features = {"joint.pos": float}
    observation_features = {"joint.pos": float}
    cameras: dict = {}

    def __init__(self):
        self.is_connected = False

    def connect(self, calibrate: bool = True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


class _FakeDataset:
    fps = 30

    def __init__(self):
        self.pending = 0
        self.num_episodes = 0
        self.saved: list[int] = []

    def has_pending_frames(self) -> bool:
        return self.pending > 0

    def add_frame(self):
        self.pending += 1

    def save_episode(self):
        if self.pending == 0:
            raise RuntimeError("save_episode on an empty buffer")
        self.saved.append(self.pending)
        self.pending = 0
        self.num_episodes += 1

    def clear_episode_buffer(self):
        self.pending = 0

    def finalize(self):
        pass


def _cfg(tmp_path, num_episodes: int) -> RecordConfig:
    return RecordConfig(
        robot=SO101FollowerConfig(port="/dev/null", id="f", calibration_dir=tmp_path),
        teleop=SO101LeaderConfig(port="/dev/null", id="l", calibration_dir=tmp_path),
        dataset=DatasetRecordConfig(
            repo_id="local/guards",
            single_task="test",
            root=tmp_path / "ds",
            num_episodes=num_episodes,
            episode_time_s=1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        play_sounds=False,
    )


def _run_record(tmp_path, num_episodes: int, episode_loop):
    """Run `record()` with fakes; `episode_loop(events, dataset)` plays one recording loop.

    Returns `(dataset, events, exit_early_at_episode_start)`.
    """
    dataset = _FakeDataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    seen_at_start: list[bool] = []

    def fake_record_loop(*, events, dataset=None, **_kwargs):
        if dataset is None:  # reset phase: no frames recorded
            return
        seen_at_start.append(events["exit_early"])
        episode_loop(events, dataset)

    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(lerobot_record, "make_robot_from_config", return_value=_FakeRobot()))
        enter(patch.object(lerobot_record, "make_teleoperator_from_config", return_value=None))
        enter(patch.object(lerobot_record.LeRobotDataset, "create", return_value=dataset))
        enter(patch.object(lerobot_record, "init_keyboard_listener", return_value=(None, events)))
        install_signals = enter(
            patch.object(
                lerobot_record,
                "install_recording_signal_handlers",
                wraps=lerobot_record.install_recording_signal_handlers,
            )
        )
        enter(patch.object(lerobot_record, "record_loop", side_effect=fake_record_loop))
        enter(patch.object(lerobot_record, "VideoEncodingManager", lambda ds: contextlib.nullcontext()))
        enter(patch.object(lerobot_record, "log_say"))
        enter(patch.object(lerobot_record, "init_logging"))
        # Helpers that only exist in some upstream versions.
        for name in ("check_control_rate", "start_episode", "CycleTimer", "init_rerun", "init_visualization"):
            if hasattr(lerobot_record, name):
                enter(patch.object(lerobot_record, name, MagicMock()))
        sigterm_before = signal.getsignal(signal.SIGTERM)
        record(_cfg(tmp_path, num_episodes))

    # The controller-app drives recording with SIGUSR1/SIGUSR2/SIGTERM (test_recording_signals.py):
    # installed on the session's events, and the previous handlers are back once record() returns.
    install_signals.assert_called_once_with(events)
    assert signal.getsignal(signal.SIGTERM) is sigterm_before
    return dataset, events, seen_at_start


def test_stale_exit_early_does_not_skip_next_episode(tmp_path):
    def episode_loop(events, dataset):
        if events["exit_early"]:
            return  # what the real loop does: leave on the first tick
        for _ in range(3):
            dataset.add_frame()
        # Operator hits "next" just as the loop finishes on its own (timer ran out) — the
        # flag outlives the loop and is still set while save_episode encodes.
        events["exit_early"] = True

    dataset, _, seen_at_start = _run_record(tmp_path, num_episodes=3, episode_loop=episode_loop)

    assert seen_at_start == [False, False, False]
    assert dataset.saved == [3, 3, 3]


def test_empty_episode_is_skipped_not_saved(tmp_path, caplog):
    calls = {"n": 0}

    def episode_loop(events, dataset):
        calls["n"] += 1
        if calls["n"] == 2:
            events["exit_early"] = True  # "next" on the very first tick: no frames
            return
        for _ in range(2):
            dataset.add_frame()

    with caplog.at_level(logging.WARNING):
        dataset, _, seen_at_start = _run_record(tmp_path, num_episodes=2, episode_loop=episode_loop)

    # Episode 1 was attempted twice: once empty (skipped), once with frames.
    assert len(seen_at_start) == 3
    assert dataset.saved == [2, 2]
    assert "No frames were recorded for this episode" in caplog.text


def test_stop_on_first_tick_ends_session_without_saving(tmp_path):
    def episode_loop(events, dataset):
        events["stop_recording"] = True
        events["exit_early"] = True

    dataset, _, seen_at_start = _run_record(tmp_path, num_episodes=5, episode_loop=episode_loop)

    assert seen_at_start == [False]
    assert dataset.saved == []
