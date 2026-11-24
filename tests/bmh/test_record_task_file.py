"""BMH: `RecordConfig.task_file` — the controller-app changes the prompt mid-session.

The app writes `{"task": "..."}` to the file; `record()` reads it at every episode start, so the
new prompt applies from the next episode and never inside one. Each read is logged as
`BMH_RECORD_TASK {"episode": i, "task": "..."}`, which the app parses to show the prompt an
episode is saved with. Without `task_file` nothing changes (upstream behaviour).
"""

import contextlib
import json
import logging
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("datasets")
pytestmark = pytest.mark.dataset

from lerobot.configs.dataset import DatasetRecordConfig  # noqa: E402
from lerobot.robots.so_follower import SO101FollowerConfig  # noqa: E402
from lerobot.scripts import lerobot_record  # noqa: E402
from lerobot.scripts.lerobot_record import RecordConfig, record  # noqa: E402
from lerobot.teleoperators.so_leader import SO101LeaderConfig  # noqa: E402
from tests.bmh.test_record_guards import _FakeDataset, _FakeRobot  # noqa: E402

FALLBACK = "dataset task"
LOG_PREFIX = "BMH_RECORD_TASK "


def _cfg(tmp_path, num_episodes: int, task_file) -> RecordConfig:
    return RecordConfig(
        robot=SO101FollowerConfig(port="/dev/null", id="f", calibration_dir=tmp_path),
        teleop=SO101LeaderConfig(port="/dev/null", id="l", calibration_dir=tmp_path),
        dataset=DatasetRecordConfig(
            repo_id="local/task-file",
            single_task=FALLBACK,
            root=tmp_path / "ds",
            num_episodes=num_episodes,
            episode_time_s=1,
            reset_time_s=0,
            push_to_hub=False,
        ),
        task_file=None if task_file is None else str(task_file),
        play_sounds=False,
    )


def _write(path, task) -> None:
    path.write_text(json.dumps({"task": task}))


def _run(tmp_path, num_episodes: int, task_file=None, episode_loop=None, reset_loop=None):
    """Run `record()` with fakes. `episode_loop(n, events)` runs inside the n-th recording loop
    (0-based, re-records count), `reset_loop(n)` inside the n-th reset phase.

    Returns `(dataset, calls)`; `calls` is one `(kind, single_task)` per `record_loop` call.
    """
    dataset = _FakeDataset()
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    calls: list[tuple[str, str]] = []
    counts = {"episode": 0, "reset": 0}

    def fake_record_loop(*, events, dataset=None, single_task=None, **_kwargs):
        kind = "reset" if dataset is None else "episode"
        calls.append((kind, single_task))
        n = counts[kind]
        counts[kind] += 1
        if kind == "reset":
            if reset_loop is not None:
                reset_loop(n)
            return
        dataset.add_frame()
        if episode_loop is not None:
            episode_loop(n, events)

    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(lerobot_record, "make_robot_from_config", return_value=_FakeRobot()))
        enter(patch.object(lerobot_record, "make_teleoperator_from_config", return_value=None))
        enter(patch.object(lerobot_record.LeRobotDataset, "create", return_value=dataset))
        enter(patch.object(lerobot_record, "init_keyboard_listener", return_value=(None, events)))
        enter(patch.object(lerobot_record, "install_recording_signal_handlers", return_value=None))
        enter(patch.object(lerobot_record, "record_loop", side_effect=fake_record_loop))
        enter(patch.object(lerobot_record, "VideoEncodingManager", lambda ds: contextlib.nullcontext()))
        enter(patch.object(lerobot_record, "log_say"))
        enter(patch.object(lerobot_record, "init_logging"))
        for name in ("check_control_rate", "start_episode", "CycleTimer", "init_rerun", "init_visualization"):
            if hasattr(lerobot_record, name):
                enter(patch.object(lerobot_record, name, MagicMock()))
        record(_cfg(tmp_path, num_episodes, task_file))
    return dataset, calls


def _episode_tasks(calls) -> list[str]:
    return [task for kind, task in calls if kind == "episode"]


def _logged(caplog) -> list[dict]:
    return [
        json.loads(r.getMessage()[len(LOG_PREFIX) :])
        for r in caplog.records
        if r.getMessage().startswith(LOG_PREFIX)
    ]


def test_without_task_file_every_episode_gets_the_dataset_task(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        dataset, calls = _run(tmp_path, num_episodes=3)

    assert dataset.saved == [1, 1, 1]
    assert {task for _, task in calls} == {FALLBACK}
    assert _logged(caplog) == []


def test_rewrite_between_episodes_applies_from_the_next_one(tmp_path):
    task_file = tmp_path / "recording-task.json"
    _write(task_file, "A")

    def reset_loop(n):
        if n == 0:
            _write(task_file, "B")

    _, calls = _run(tmp_path, num_episodes=3, task_file=task_file, reset_loop=reset_loop)

    # The reset phase after episode 0 still runs with episode 0's task.
    assert calls == [("episode", "A"), ("reset", "A"), ("episode", "B"), ("reset", "B"), ("episode", "B")]


def test_rewrite_during_an_episode_does_not_change_it(tmp_path):
    task_file = tmp_path / "recording-task.json"
    _write(task_file, "A")

    def episode_loop(n, events):
        if n == 0:
            _write(task_file, "B")

    _, calls = _run(tmp_path, num_episodes=2, task_file=task_file, episode_loop=episode_loop)

    assert _episode_tasks(calls) == ["A", "B"]


def test_rerecord_rereads_the_file(tmp_path):
    """Apply + Re-record fixes the prompt of the episode just recorded."""
    task_file = tmp_path / "recording-task.json"
    _write(task_file, "A")

    def episode_loop(n, events):
        if n == 0:
            _write(task_file, "B")
            events["rerecord_episode"] = True

    dataset, calls = _run(tmp_path, num_episodes=2, task_file=task_file, episode_loop=episode_loop)

    assert _episode_tasks(calls) == ["A", "B", "B"]
    assert dataset.saved == [1, 1]


def test_task_is_stripped(tmp_path):
    task_file = tmp_path / "recording-task.json"
    _write(task_file, "  Put the cube in the bowl \n")

    _, calls = _run(tmp_path, num_episodes=1, task_file=task_file)

    assert _episode_tasks(calls) == ["Put the cube in the bowl"]


def test_missing_file_falls_back_to_the_dataset_task(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _, calls = _run(tmp_path, num_episodes=2, task_file=tmp_path / "absent.json")

    assert _episode_tasks(calls) == [FALLBACK, FALLBACK]
    assert "task_file" not in caplog.text  # a missing file is silent


@pytest.mark.parametrize(
    "bad",
    [
        None,  # file deleted
        "{not json",
        "[1, 2]",
        '{"task": "   "}',
        '{"task": 3}',
        '{"prompt": "B"}',
    ],
    ids=["missing", "malformed", "not-object", "empty", "not-string", "no-task-key"],
)
def test_unusable_file_keeps_the_previous_task(tmp_path, caplog, bad):
    task_file = tmp_path / "recording-task.json"
    _write(task_file, "A")

    def reset_loop(n):
        if n == 0:
            if bad is None:
                task_file.unlink()
            else:
                task_file.write_text(bad)

    with caplog.at_level(logging.WARNING):
        _, calls = _run(tmp_path, num_episodes=2, task_file=task_file, reset_loop=reset_loop)

    assert _episode_tasks(calls) == ["A", "A"]
    warned = "keeping the previous task" in caplog.text
    assert warned is (bad is not None)


def test_log_line_carries_episode_index_and_task_as_json(tmp_path, caplog):
    task_file = tmp_path / "recording-task.json"
    quoted = 'Put the "red" cube in the bowl\\'
    _write(task_file, quoted)

    def episode_loop(n, events):
        if n == 1:
            events["rerecord_episode"] = True

    with caplog.at_level(logging.INFO):
        _run(tmp_path, num_episodes=2, task_file=task_file, episode_loop=episode_loop)

    # Episode 1 is re-recorded, so it is announced twice.
    assert _logged(caplog) == [
        {"episode": 0, "task": quoted},
        {"episode": 1, "task": quoted},
        {"episode": 1, "task": quoted},
    ]
    # One line each: the app reads the log line by line.
    assert all("\n" not in r.getMessage() for r in caplog.records if r.getMessage().startswith(LOG_PREFIX))
