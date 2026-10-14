"""BMH: temporary frames of `video` features are written as JPEG, not PNG (Pi CPU fix).

PNG at compress level 1 cost ~25 ms per 640x480 frame on a Pi 5 — two cores at 3 cameras x 30 fps.
JPEG at quality 95 costs ~2 ms. Permanent `image` features must stay lossless PNG.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lerobot.datasets.image_writer import VIDEO_FRAME_JPEG_QUALITY, write_image
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH, DEFAULT_VIDEO_FRAME_PATH
from lerobot.datasets.video_utils import encode_video_frames


def _frame(h: int = 48, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_video_frame_path_is_jpg_and_image_path_stays_png():
    video = DEFAULT_VIDEO_FRAME_PATH.format(image_key="cam", episode_index=3, frame_index=7)
    image = DEFAULT_IMAGE_PATH.format(image_key="cam", episode_index=3, frame_index=7)
    assert video.endswith("frame-000007.jpg")
    assert image.endswith("frame-000007.png")
    # Same directory: episode cleanup / encode workers locate frames by parent dir.
    assert Path(video).parent == Path(image).parent


@pytest.mark.parametrize("suffix,fmt", [(".jpg", "JPEG"), (".jpeg", "JPEG"), (".png", "PNG")])
def test_write_image_picks_format_from_suffix(tmp_path, suffix, fmt):
    fpath = tmp_path / f"frame-000000{suffix}"
    write_image(_frame(), fpath, compress_level=1)
    with Image.open(fpath) as img:
        assert img.format == fmt
        assert img.size == (64, 48)
        assert img.mode == "RGB"


def test_jpeg_frame_is_close_to_source(tmp_path):
    # Smooth content, like a camera frame — pure noise is a meaningless JPEG benchmark.
    yy, xx = np.mgrid[0:48, 0:64]
    src = np.stack([xx * 4, yy * 5, (xx + yy) * 2], axis=-1).astype(np.uint8)
    jpg = tmp_path / "f.jpg"
    write_image(src, jpg)
    with Image.open(jpg) as img:
        back = np.asarray(img.convert("RGB")).astype(np.int16)
    assert np.abs(back - src.astype(np.int16)).mean() < 2.0, "quality 95 should be near-lossless"
    assert VIDEO_FRAME_JPEG_QUALITY >= 90


def test_encode_video_frames_accepts_jpg_frames(tmp_path):
    pytest.importorskip("av")
    imgs_dir = tmp_path / "frames"
    imgs_dir.mkdir()
    n = 6
    for i in range(n):
        write_image(_frame(seed=i), imgs_dir / f"frame-{i:06d}.jpg")
    out = tmp_path / "out.mp4"
    encode_video_frames(imgs_dir, out, fps=30, vcodec="h264", g=2, crf=23)
    assert out.exists() and out.stat().st_size > 0


def test_encode_video_frames_orders_mixed_png_and_jpg_by_index(tmp_path, monkeypatch):
    """Both suffixes are globbed and sorted by frame index, not by extension."""
    import lerobot.datasets.video_utils as vu

    imgs_dir = tmp_path / "frames"
    imgs_dir.mkdir()
    write_image(_frame(seed=1), imgs_dir / "frame-000001.jpg")
    write_image(_frame(seed=0), imgs_dir / "frame-000000.png")
    write_image(_frame(seed=2), imgs_dir / "frame-000002.jpg")

    opened: list[str] = []
    real_open = vu.Image.open

    def spy_open(path, *a, **k):
        opened.append(Path(path).name)
        return real_open(path, *a, **k)

    monkeypatch.setattr(vu.Image, "open", spy_open)

    class _Stream:
        pix_fmt = width = height = None

        def encode(self, *_):
            return None

    class _Out:
        def add_stream(self, *_, **__):
            return _Stream()

        def mux(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_av_open(path, mode):
        Path(path).write_bytes(b"x")
        return _Out()

    monkeypatch.setattr(vu.av, "open", fake_av_open)
    monkeypatch.setattr(vu.av, "VideoFrame", type("VF", (), {"from_image": staticmethod(lambda img: img)}))
    encode_video_frames(imgs_dir, tmp_path / "out.mp4", fps=30)
    # First open is the size probe on frame 0, then the encode loop in index order.
    assert opened == ["frame-000000.png", "frame-000000.png", "frame-000001.jpg", "frame-000002.jpg"]


def test_dataset_writer_uses_jpg_for_video_and_png_for_image(tmp_path):
    pytest.importorskip("datasets")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.cam": {"dtype": "video", "shape": (48, 64, 3), "names": ["h", "w", "c"]},
        "observation.still": {"dtype": "image", "shape": (48, 64, 3), "names": ["h", "w", "c"]},
        "state": {"dtype": "float32", "shape": (2,), "names": None},
    }
    ds = LeRobotDataset.create(
        repo_id="bmh/jpeg-frames", fps=30, features=features, root=tmp_path / "ds", image_writer_threads=0
    )
    writer = ds.writer
    frame = {
        "task": "t",
        "observation.images.cam": _frame(),
        "observation.still": _frame(seed=1),
        "state": np.zeros(2, dtype=np.float32),
    }
    ds.add_frame(frame)
    cam_path = Path(writer.episode_buffer["observation.images.cam"][0])
    still_path = Path(writer.episode_buffer["observation.still"][0])
    assert cam_path.suffix == ".jpg" and cam_path.exists()
    assert still_path.suffix == ".png" and still_path.exists()
    with Image.open(cam_path) as img:
        assert img.format == "JPEG"
    with Image.open(still_path) as img:
        assert img.format == "PNG"
    # Cleanup helpers resolve the per-key directory through the same template switch.
    assert writer._get_image_file_dir(0, "observation.images.cam") == cam_path.parent
    writer.cleanup_interrupted_episode(0)
    assert not cam_path.exists() and not still_path.exists()
