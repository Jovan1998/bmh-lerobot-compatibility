#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""BMH: low-rate JPEG preview tapped from an OpenCV camera's read thread.

The controller-app shows a live view of cameras that `lerobot-record` holds open exclusively by
reading the JPEG snapshots the camera's read thread writes to tmpfs.
"""

import logging
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.preview import FramePreviewWriter


def _bgr(h: int = 48, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class _Clock:
    """Stand-in for `time.perf_counter` inside the preview module."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    c = _Clock()
    with patch("lerobot.cameras.preview.time.perf_counter", c):
        yield c


# --- FramePreviewWriter ------------------------------------------------------------------


def test_first_frame_is_written_as_jpeg_at_preview_width(tmp_path):
    path = tmp_path / "preview.jpg"
    writer = FramePreviewWriter(str(path), fps=3, width=32, quality=60)

    writer.maybe_write(_bgr(48, 64))

    img = cv2.imread(str(path))
    assert img is not None
    assert path.read_bytes()[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert img.shape[1] == 32
    assert img.shape[0] == 24  # aspect kept


def test_narrow_frames_are_not_upscaled(tmp_path):
    path = tmp_path / "preview.jpg"
    FramePreviewWriter(str(path), fps=3, width=320, quality=60).maybe_write(_bgr(48, 64))
    assert cv2.imread(str(path)).shape[:2] == (48, 64)


def test_rotation_is_applied_before_encoding(tmp_path):
    path = tmp_path / "preview.jpg"
    writer = FramePreviewWriter(str(path), fps=3, width=320, quality=60, rotation=cv2.ROTATE_90_CLOCKWISE)
    writer.maybe_write(_bgr(48, 64))
    assert cv2.imread(str(path)).shape[:2] == (64, 48)


def test_writes_at_most_fps_snapshots_per_second(tmp_path, clock):
    path = tmp_path / "preview.jpg"
    writer = FramePreviewWriter(str(path), fps=4, width=64, quality=60)

    with patch("lerobot.cameras.preview.os.replace", wraps=os.replace) as replace:
        # 30 fps camera for one second → 4 snapshots (t = 0, 0.25, 0.5, 0.75 s).
        for i in range(30):
            clock.t = 1000.0 + i / 30
            writer.maybe_write(_bgr())
        assert replace.call_count == 4

        # Inside the interval: nothing; at the interval: one more.
        clock.t = 1000.0 + 29 / 30 + 0.01
        writer.maybe_write(_bgr())
        assert replace.call_count == 4
        clock.t += 0.25
        writer.maybe_write(_bgr())
        assert replace.call_count == 5


def test_file_is_replaced_atomically(tmp_path, clock):
    path = tmp_path / "preview.jpg"
    path.write_bytes(b"previous snapshot")
    writer = FramePreviewWriter(str(path), fps=3, width=64, quality=60)
    seen: dict = {}
    real_replace = os.replace

    def _checking_replace(src, dst):
        # At the moment of the swap the reader still sees the old, complete file and the new
        # snapshot is fully written next to it.
        with open(dst, "rb") as f:
            seen["dst_before"] = f.read()
        seen["src_is_jpeg"] = cv2.imread(str(src)) is not None
        seen["src"], seen["dst"] = str(src), str(dst)
        real_replace(src, dst)

    with patch("lerobot.cameras.preview.os.replace", side_effect=_checking_replace):
        writer.maybe_write(_bgr())

    assert seen["dst_before"] == b"previous snapshot"
    assert seen["src_is_jpeg"]
    assert seen["dst"] == str(path)
    assert seen["src"] != seen["dst"]
    assert not os.path.exists(seen["src"])  # tmp file consumed by the rename
    assert cv2.imread(str(path)) is not None


def test_close_removes_the_file_and_recreates_dir_on_next_write(tmp_path, clock):
    path = tmp_path / "sub" / "preview.jpg"
    writer = FramePreviewWriter(str(path), fps=3, width=64, quality=60)
    writer.maybe_write(_bgr())
    assert path.exists()

    writer.close()
    assert not path.exists()
    writer.close()  # idempotent

    (tmp_path / "sub").rmdir()
    clock.t += 1.0
    writer.maybe_write(_bgr())
    assert path.exists()


def test_write_failure_is_swallowed_and_logged_once(tmp_path, clock, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    writer = FramePreviewWriter(str(blocker / "preview.jpg"), fps=3, width=64, quality=60)

    with caplog.at_level(logging.WARNING, logger="lerobot.cameras.preview"):
        for _ in range(3):
            clock.t += 1.0
            writer.maybe_write(_bgr())  # must not raise — preview never interrupts recording

    assert caplog.text.count("Preview write") == 1


# --- OpenCVCameraConfig ------------------------------------------------------------------


def test_preview_disabled_by_default():
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=64, height=48, fps=30))
    assert cam._preview is None


@pytest.mark.parametrize(
    "kwargs", [{"preview_fps": 0}, {"preview_quality": 0}, {"preview_quality": 101}, {"preview_width": 0}]
)
def test_invalid_preview_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OpenCVCameraConfig(index_or_path=0, preview_path="/tmp/x.jpg", **kwargs)


# --- OpenCVCamera read loop tap ----------------------------------------------------------


def test_read_loop_feeds_raw_frames_to_preview_and_disconnect_closes_it(tmp_path):
    path = tmp_path / "preview-front.jpg"
    cfg = OpenCVCameraConfig(
        index_or_path=0,
        width=64,
        height=48,
        fps=30,
        rotation=Cv2Rotation.NO_ROTATION,
        preview_path=str(path),
        preview_fps=30,
        preview_width=32,
    )
    cam = OpenCVCamera(cfg)
    assert isinstance(cam._preview, FramePreviewWriter)
    raw = _bgr(48, 64)

    with (
        patch.object(cam, "_read_from_hardware", return_value=raw),
        patch.object(cam._preview, "maybe_write", wraps=cam._preview.maybe_write) as maybe_write,
        patch.object(cam._preview, "close", wraps=cam._preview.close) as close,
    ):
        cam._start_read_thread()
        assert cam.new_frame_event.wait(timeout=2.0)
        assert maybe_write.call_count >= 1
        # The preview gets the raw BGR frame (what cv2.imencode expects), not the RGB-converted one.
        assert maybe_write.call_args.args[0] is raw
        assert path.exists()

        cam.disconnect()

    close.assert_called_once()
    assert not path.exists()
    assert cam.thread is None
