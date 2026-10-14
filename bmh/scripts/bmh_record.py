#!/usr/bin/env python
"""bmh-record — `lerobot-record` driven by a JSON config file instead of CLI flags.

`lerobot-record` parses its flags with draccus, which builds an argparse tree over every
registered robot, teleoperator and policy config on every start. On a Raspberry Pi 5 that
alone costs ~5.5 s before the robot is even touched. The controller-app already knows the
full configuration, so it writes it to a JSON file and this entry decodes it in a few
milliseconds with `draccus.decode` — the same decoder draccus uses for `--config_path`
files, so the resulting `RecordConfig` is identical to what the CLI would have produced
(verified field-by-field on the Pi).

`lerobot.configs.parser.wrap` skips the CLI parse when the wrapped function is handed a
ready `RecordConfig`, so `record(cfg)` runs the unmodified recording flow.

The JSON mirrors the CLI structure. ChoiceRegistry fields carry a `type` key, exactly like
the camera dicts on the `lerobot-record` command line:

    {
      "robot": {"type": "bi_so_follower", "id": "so_follower",
                "left_arm_config": {"port": "/dev/ttyACM1", "with_head": true,
                                    "cameras": {"front": {"type": "opencv", "index_or_path": 0, ...}}},
                "right_arm_config": {"port": "/dev/ttyACM0", "cameras": {...}}},
      "teleop": {"type": "bi_so_network_leader", "left_arm_config": {...}, "right_arm_config": {...}},
      "dataset": {"repo_id": "local/my-set", "root": "/path/to/staging", "single_task": "...", ...},
      "resume": false,
      "play_sounds": false
    }

Usage:
    bmh-record --config ~/.cache/bmh-101/recording-config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_record_config(path: Path | str) -> Any:
    """Decode a JSON file into a `RecordConfig` (imports lerobot lazily)."""
    import draccus
    from draccus.utils import ParsingError

    from lerobot.scripts.lerobot_record import RecordConfig

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at the top level, got {type(data).__name__}")
    try:
        return draccus.decode(RecordConfig, data)
    except ParsingError as e:
        # draccus hides the dataclass' own error (e.g. RecordConfig.__post_init__'s
        # "A teleoperator is required") behind a generic message; surface it.
        cause = f" ({e.__cause__})" if e.__cause__ else ""
        raise ValueError(f"{path}: {e}{cause}") from e


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bmh-record",
        description="Run lerobot-record from a JSON config file (skips the slow draccus CLI parse).",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to the RecordConfig JSON file")
    args = parser.parse_args(argv)

    # Heavy imports only after argument errors have been reported.
    from lerobot.scripts.lerobot_record import record
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()  # parity with `lerobot-record`; a few ms
    cfg = load_record_config(args.config)
    record(cfg)


if __name__ == "__main__":
    main()
