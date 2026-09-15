#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Contract tests for DatasetWriter."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.dataset_writer import _encode_video_worker
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from tests.fixtures.constants import DEFAULT_FPS, DUMMY_REPO_ID

SIMPLE_FEATURES = {
    "state": {"dtype": "float32", "shape": (6,), "names": None},
    "action": {"dtype": "float32", "shape": (6,), "names": None},
}


def _make_frame(features: dict, task: str = "Dummy task") -> dict:
    """Build a valid frame dict for the given features."""
    frame = {"task": task}
    for key, ft in features.items():
        if ft["dtype"] in ("image", "video"):
            frame[key] = np.random.randint(0, 256, size=ft["shape"], dtype=np.uint8)
        elif ft["dtype"] in ("float32", "float64"):
            frame[key] = torch.randn(ft["shape"])
        elif ft["dtype"] == "int64":
            frame[key] = torch.zeros(ft["shape"], dtype=torch.int64)
    return frame


# ── Existing encode_video_worker tests ───────────────────────────────


def test_encode_video_worker_forwards_vcodec(tmp_path):
    """_encode_video_worker correctly forwards the vcodec parameter."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30, vcodec="h264")

    assert captured_kwargs["vcodec"] == "h264"


def test_encode_video_worker_default_vcodec(tmp_path):
    """_encode_video_worker uses libsvtav1 as the default codec."""
    video_key = "observation.images.laptop"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=0, frame_index=0)
    img_dir = tmp_path / Path(fpath).parent
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color="red").save(img_dir / "frame-000000.png")

    captured_kwargs = {}

    def mock_encode(imgs_dir, video_path, fps, **kwargs):
        captured_kwargs.update(kwargs)
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        Path(video_path).touch()

    with patch("lerobot.datasets.dataset_writer.encode_video_frames", side_effect=mock_encode):
        _encode_video_worker(video_key, 0, tmp_path, fps=30)

    assert captured_kwargs["vcodec"] == "libsvtav1"


# ── add_frame contracts ──────────────────────────────────────────────


def test_add_frame_increments_buffer_size(tmp_path):
    """Each add_frame() call increases episode_buffer['size'] by 1."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    assert dataset.writer.episode_buffer["size"] == 0

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 2


def test_add_frame_rejects_missing_feature(tmp_path):
    """add_frame() raises ValueError when a required feature is missing."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    with pytest.raises(ValueError, match="Missing features"):
        dataset.add_frame({"task": "Dummy task", "state": torch.randn(6)})
        # missing 'action'


# ── save_episode contracts ───────────────────────────────────────────


def test_save_episode_writes_parquet(tmp_path):
    """After save_episode(), at least one .parquet file exists under data/."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    parquet_files = list((tmp_path / "ds" / "data").rglob("*.parquet"))
    assert len(parquet_files) > 0


def test_save_episode_updates_counters(tmp_path):
    """After save_episode(), metadata counters are updated."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(5):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 5


def test_save_episode_resets_buffer(tmp_path):
    """After save_episode(), the episode buffer is reset."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    assert dataset.writer.episode_buffer["size"] == 0


def test_save_multiple_episodes(tmp_path):
    """Recording 3 episodes results in correct total counts."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    total_frames = 0
    for ep in range(3):
        n_frames = ep + 2  # 2, 3, 4
        for _ in range(n_frames):
            dataset.add_frame(_make_frame(SIMPLE_FEATURES))
        dataset.save_episode()
        total_frames += n_frames

    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == total_frames


# ── clear / lifecycle ────────────────────────────────────────────────


def test_clear_resets_buffer(tmp_path):
    """clear_episode_buffer() resets the buffer size to 0."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    assert dataset.writer.episode_buffer["size"] == 1

    dataset.clear_episode_buffer()
    assert dataset.writer.episode_buffer["size"] == 0


def test_finalize_is_idempotent(tmp_path):
    """Calling finalize() twice does not raise."""
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=tmp_path / "ds"
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    dataset.finalize()
    dataset.finalize()  # second call should not raise


def test_finalize_then_read_roundtrip(tmp_path):
    """Write data, finalize, re-open, and verify data matches."""
    root = tmp_path / "roundtrip"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=features, root=root)

    # Record known values
    known_states = []
    for i in range(5):
        state = torch.tensor([float(i), float(i * 10)])
        known_states.append(state)
        dataset.add_frame({"task": "Test task", "state": state})
    dataset.save_episode()
    dataset.finalize()

    # Read back
    for i in range(5):
        item = dataset[i]
        assert torch.allclose(item["state"], known_states[i], atol=1e-5)


# ── durability: the data parquet is closed before slow video work ────


VIDEO_FEATURES = {
    **SIMPLE_FEATURES,
    "observation.images.cam": {
        "dtype": "video",
        "shape": (8, 8, 3),
        "names": ["height", "width", "channels"],
    },
}


def _data_parquet_files(root: Path) -> list[Path]:
    return sorted((root / "data").rglob("*.parquet"))


def _rows_on_disk(root: Path) -> int:
    """Row count across all data files; raises if any file lacks its footer."""
    return sum(pq.read_table(f).num_rows for f in _data_parquet_files(root))


def test_close_writer_rolls_to_new_data_file(tmp_path):
    """close_writer() between episodes is a checkpoint: the closed file is left
    untouched and the next episode goes to a fresh data file."""
    root = tmp_path / "ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()
    dataset.writer.close_writer()

    (first_file,) = _data_parquet_files(root)
    first_bytes = first_file.read_bytes()
    assert pq.read_table(first_file).num_rows == 3

    for _ in range(2):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()
    dataset.finalize()

    files = _data_parquet_files(root)
    assert len(files) == 2
    assert files[0].read_bytes() == first_bytes
    assert pq.read_table(files[1]).num_rows == 2
    assert dataset.meta.total_episodes == 2
    assert dataset.meta.total_frames == 5
    assert dataset[4]["state"].shape == (6,)


def test_finalize_closes_parquet_before_video_flush(tmp_path):
    """finalize() writes the data parquet footer *before* the (slow) video flush,
    so a kill during encoding cannot corrupt the recorded rows."""
    root = tmp_path / "ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=DEFAULT_FPS, features=SIMPLE_FEATURES, root=root
    )
    for _ in range(3):
        dataset.add_frame(_make_frame(SIMPLE_FEATURES))
    dataset.save_episode()

    rows_at_flush: list[int] = []
    with patch.object(
        dataset.writer,
        "flush_pending_videos",
        side_effect=lambda: rows_at_flush.append(_rows_on_disk(root)),
    ):
        dataset.finalize()

    assert rows_at_flush == [3]


def test_batch_encode_checkpoints_parquet_before_encoding(tmp_path):
    """At a batch-encode boundary the data parquet is closed (footer written) before
    any video is encoded, and recording continues into a new data file afterwards."""
    root = tmp_path / "ds"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=DEFAULT_FPS,
        features=VIDEO_FEATURES,
        root=root,
        batch_encoding_size=2,
    )
    rows_at_encode: list[int] = []

    def fake_save_episode_video(video_key: str, episode_index: int, temp_path=None) -> dict:
        rows_at_encode.append(_rows_on_disk(root))
        return {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
            f"videos/{video_key}/from_timestamp": 0.0,
            f"videos/{video_key}/to_timestamp": 0.1,
        }

    with patch.object(dataset.writer, "_save_episode_video", side_effect=fake_save_episode_video):
        for _ in range(3):
            for _ in range(3):
                dataset.add_frame(_make_frame(VIDEO_FEATURES))
            dataset.save_episode()
        # Batch boundary after episode 2: both episodes' rows were on disk before encoding.
        assert rows_at_encode == [6, 6]
        dataset.finalize()

    # finalize() checkpointed again before encoding the leftover third episode.
    assert rows_at_encode == [6, 6, 9]
    assert len(_data_parquet_files(root)) == 2
    assert dataset.meta.total_episodes == 3
    assert dataset.meta.total_frames == 9
