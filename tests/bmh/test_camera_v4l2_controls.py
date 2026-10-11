"""Unit tests for the BMH ``OpenCVCamera`` extras — no camera, no ``v4l2-ctl``.

Covers ``v4l2_controls`` (``_apply_v4l2_controls``: command lines, readback verification, platform
and missing-tool handling) and the frame-rate watchdog in ``_read_loop``. Run with
``uv run --no-sync --with msgpack-numpy pytest tests/bmh/test_camera_v4l2_controls.py -q``.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest

from lerobot.cameras.opencv import camera_opencv
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, _parse_v4l2_controls
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# --------------------------------------------------------------------------- helpers


class FakeRun:
    """Stands in for ``subprocess.run``: records argv, answers ``-C`` with canned readback output."""

    def __init__(self, readback: str = "", missing: bool = False):
        self.readback = readback
        self.missing = missing
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        if self.missing:
            raise FileNotFoundError(2, "No such file or directory", "v4l2-ctl")
        self.calls.append(list(args))
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        stdout = self.readback if "-C" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


def _camera(index_or_path, controls) -> OpenCVCamera:
    return OpenCVCamera(
        OpenCVCameraConfig(index_or_path=index_or_path, width=640, height=480, fps=30, v4l2_controls=controls)
    )


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")


# --------------------------------------------------------------------------- config


def test_config_defaults_to_none():
    assert OpenCVCameraConfig(index_or_path=0).v4l2_controls is None


# --------------------------------------------------------------------------- readback parser


def test_parse_v4l2_controls_output():
    out = (
        "exposure_dynamic_framerate: 0\nauto_exposure: 3 (Aperture Priority Mode)\n\ngarbage line\nfoo: bar\n"
    )
    assert _parse_v4l2_controls(out) == {"exposure_dynamic_framerate": 0, "auto_exposure": 3}


# --------------------------------------------------------------------------- _apply_v4l2_controls


def test_applies_and_verifies_control_for_int_index(linux, monkeypatch, caplog):
    run = FakeRun(readback="exposure_dynamic_framerate: 0\n")
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)
    cam = _camera(4, {"exposure_dynamic_framerate": 0})

    with caplog.at_level(logging.INFO, logger=camera_opencv.__name__):
        cam._apply_v4l2_controls()

    assert run.calls == [
        ["v4l2-ctl", "-d", "/dev/video4", "-c", "exposure_dynamic_framerate=0"],
        ["v4l2-ctl", "-d", "/dev/video4", "-C", "exposure_dynamic_framerate"],
    ]
    assert "v4l2 controls applied: {'exposure_dynamic_framerate': 0}" in caplog.text


def test_uses_device_path_and_joins_multiple_controls(linux, monkeypatch):
    run = FakeRun(readback="auto_exposure: 1 (Manual Mode)\nexposure_time_absolute: 300\n")
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)
    cam = _camera(
        Path("/dev/v4l/by-id/usb-Innomaker-video-index0"), {"auto_exposure": 1, "exposure_time_absolute": 300}
    )

    cam._apply_v4l2_controls()

    dev = "/dev/v4l/by-id/usb-Innomaker-video-index0"
    assert run.calls == [
        ["v4l2-ctl", "-d", dev, "-c", "auto_exposure=1,exposure_time_absolute=300"],
        ["v4l2-ctl", "-d", dev, "-C", "auto_exposure,exposure_time_absolute"],
    ]


def test_readback_mismatch_raises(linux, monkeypatch):
    run = FakeRun(readback="exposure_dynamic_framerate: 1\n")
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)
    cam = _camera(4, {"exposure_dynamic_framerate": 0})

    with pytest.raises(RuntimeError, match=r"exposure_dynamic_framerate=0 \(device reports 1\)"):
        cam._apply_v4l2_controls()


def test_missing_control_in_readback_raises(linux, monkeypatch):
    run = FakeRun(readback="")
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)
    cam = _camera(4, {"exposure_dynamic_framerate": 0})

    with pytest.raises(RuntimeError, match=r"device reports None"):
        cam._apply_v4l2_controls()


def test_v4l2_ctl_failure_raises(linux, monkeypatch):
    def failing_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="unknown control 'bogus'\n")

    monkeypatch.setattr(camera_opencv.subprocess, "run", failing_run)
    cam = _camera(4, {"bogus": 0})

    with pytest.raises(RuntimeError, match=r"unknown control 'bogus'"):
        cam._apply_v4l2_controls()


def test_missing_v4l2_ctl_raises_with_hint(linux, monkeypatch):
    monkeypatch.setattr(camera_opencv.subprocess, "run", FakeRun(missing=True))
    cam = _camera(4, {"exposure_dynamic_framerate": 0})

    with pytest.raises(RuntimeError, match=r"v4l-utils"):
        cam._apply_v4l2_controls()


@pytest.mark.parametrize("controls", [None, {}])
def test_no_subprocess_when_unset(linux, monkeypatch, controls):
    run = FakeRun()
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)

    _camera(4, controls)._apply_v4l2_controls()

    assert run.calls == []


@pytest.mark.parametrize("plat", ["darwin", "win32"])
def test_no_subprocess_off_linux(monkeypatch, plat):
    monkeypatch.setattr(sys, "platform", plat)
    run = FakeRun()
    monkeypatch.setattr(camera_opencv.subprocess, "run", run)

    _camera(4, {"exposure_dynamic_framerate": 0})._apply_v4l2_controls()

    assert run.calls == []


# --------------------------------------------------------------------------- fps watchdog


def _drive_read_loop(monkeypatch, frame_period_s: float, duration_s: float, fps: int = 30) -> OpenCVCamera:
    """Runs ``_read_loop`` for ``duration_s`` against a fake camera emitting a frame every ``frame_period_s``."""
    cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=64, height=48, fps=fps))
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    def fake_read():
        time.sleep(frame_period_s)
        return frame

    monkeypatch.setattr(cam, "_read_from_hardware", fake_read)
    cam.stop_event = Event()
    thread = Thread(target=cam._read_loop, daemon=True)
    thread.start()
    time.sleep(duration_s)
    cam.stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert cam.latest_frame is not None
    return cam


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "delivering" in r.getMessage()]


def test_watchdog_warns_once_when_rate_too_low(monkeypatch, caplog):
    monkeypatch.setattr(camera_opencv, "FPS_WATCHDOG_WINDOW_S", 0.2)
    with caplog.at_level(logging.WARNING, logger=camera_opencv.__name__):
        # ~15 fps against configured 30; several windows pass but the 60 s rate limit allows one warning.
        _drive_read_loop(monkeypatch, frame_period_s=0.066, duration_s=1.0)

    records = _warnings(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert msg.startswith("OpenCVCamera(0) delivering ")
    assert msg.endswith(" fps, configured 30")
    assert float(msg.split("delivering ")[1].split(" fps")[0]) < 27


def test_watchdog_rate_limit_allows_another_warning_after_interval(monkeypatch, caplog):
    monkeypatch.setattr(camera_opencv, "FPS_WATCHDOG_WINDOW_S", 0.2)
    monkeypatch.setattr(camera_opencv, "FPS_WATCHDOG_WARN_INTERVAL_S", 0.5)
    with caplog.at_level(logging.WARNING, logger=camera_opencv.__name__):
        _drive_read_loop(monkeypatch, frame_period_s=0.066, duration_s=1.3)

    assert len(_warnings(caplog)) == 2


def test_watchdog_silent_at_configured_rate(monkeypatch, caplog):
    monkeypatch.setattr(camera_opencv, "FPS_WATCHDOG_WINDOW_S", 0.2)
    with caplog.at_level(logging.WARNING, logger=camera_opencv.__name__):
        _drive_read_loop(monkeypatch, frame_period_s=0.01, duration_s=0.7)

    assert _warnings(caplog) == []
