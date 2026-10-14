"""BMH: `bmh-record` builds the same RecordConfig from JSON that `lerobot-record` parses from flags."""

import json
from unittest.mock import patch

import pytest

# `lerobot.scripts.lerobot_record` needs the `datasets` extra (not in the Mac dev venv; the Pi has it).
pytest.importorskip("datasets")

from bmh.scripts.bmh_record import load_record_config, main  # noqa: E402
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollowerConfig
from lerobot.scripts.lerobot_record import RecordConfig
from lerobot.teleoperators.bi_so_network_leader import BiSONetworkLeaderConfig

CONFIG = {
    "robot": {
        "type": "bi_so_follower",
        "id": "so_follower",
        "left_arm_config": {
            "port": "/dev/ttyACM1",
            "with_head": True,
            "cameras": {
                "front": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "width": 640,
                    "height": 360,
                    "fps": 30,
                    "fourcc": "MJPG",
                    "warmup_s": 2,
                    "preview_path": "/dev/shm/bmh-101/preview-front.jpg",
                    "preview_fps": 3,
                },
                "left_wrist": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
            },
        },
        "right_arm_config": {
            "port": "/dev/ttyACM0",
            "cameras": {
                "right_wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30}
            },
        },
    },
    "teleop": {
        "type": "bi_so_network_leader",
        "left_arm_config": {
            "remote_ip": "192.168.1.2",
            "port_zmq": 5555,
            "with_head": True,
            "connect_timeout_s": 15,
        },
        "right_arm_config": {"remote_ip": "192.168.1.2", "port_zmq": 5557, "connect_timeout_s": 15},
        "lock_file": "/home/pi/.cache/bmh-101/teleop-locks.json",
    },
    "dataset": {
        "repo_id": "local/my-set",
        "root": "/tmp/staging",
        "single_task": "Place the toast",
        "num_episodes": 50,
        "episode_time_s": 600,
        "encoder_threads": 4,
        "video_encoding_batch_size": 50,
        "crf": 25,
        "g": 6,
        "push_to_hub": False,
    },
    "resume": False,
    "play_sounds": False,
}


@pytest.fixture
def config_path(tmp_path):
    p = tmp_path / "recording-config.json"
    p.write_text(json.dumps(CONFIG))
    return p


def test_load_record_config_decodes_registry_types(config_path):
    cfg = load_record_config(config_path)

    assert isinstance(cfg, RecordConfig)
    assert isinstance(cfg.robot, BiSOFollowerConfig)
    assert cfg.robot.id == "so_follower"
    assert cfg.robot.left_arm_config.port == "/dev/ttyACM1"
    assert cfg.robot.left_arm_config.with_head is True
    assert cfg.robot.right_arm_config.with_head is False
    front = cfg.robot.left_arm_config.cameras["front"]
    assert isinstance(front, OpenCVCameraConfig)
    assert (front.width, front.height, front.warmup_s, front.fourcc) == (640, 360, 2, "MJPG")
    assert front.preview_path == "/dev/shm/bmh-101/preview-front.jpg"
    assert set(cfg.robot.right_arm_config.cameras) == {"right_wrist"}

    assert isinstance(cfg.teleop, BiSONetworkLeaderConfig)
    assert cfg.teleop.left_arm_config.connect_timeout_s == 15
    assert cfg.teleop.right_arm_config.port_zmq == 5557
    assert cfg.teleop.lock_file == "/home/pi/.cache/bmh-101/teleop-locks.json"

    assert cfg.dataset.repo_id == "local/my-set"
    assert cfg.dataset.single_task == "Place the toast"
    assert cfg.dataset.video_encoding_batch_size == 50
    assert cfg.dataset.crf == 25 and cfg.dataset.g == 6
    # omitted keys keep lerobot's defaults
    assert cfg.dataset.vcodec == "libsvtav1" and cfg.dataset.preset is None
    assert cfg.resume is False and cfg.play_sounds is False


def test_load_record_config_rejects_non_object(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        load_record_config(p)


def test_load_record_config_requires_teleop(tmp_path):
    data = {k: v for k, v in CONFIG.items() if k != "teleop"}
    p = tmp_path / "no-teleop.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="teleoperator is required"):
        load_record_config(p)


def test_main_hands_decoded_config_to_record(config_path):
    with (
        patch("lerobot.scripts.lerobot_record.record") as record,
        patch("lerobot.utils.import_utils.register_third_party_plugins") as plugins,
    ):
        main(["--config", str(config_path)])
    plugins.assert_called_once_with()
    record.assert_called_once()
    (cfg,), kwargs = record.call_args
    assert kwargs == {}
    assert isinstance(cfg, RecordConfig) and cfg.dataset.repo_id == "local/my-set"


def test_main_requires_config_flag():
    with pytest.raises(SystemExit):
        main([])
