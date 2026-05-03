# Follower Server — Network Teleoperator

This directory contains the **follower-side** network teleoperation modules.
These are lerobot `Teleoperator` implementations that receive leader arm positions
over ZMQ instead of reading from local USB.

## Modules

### `so_network_leader/`
Drop-in replacement for `SOLeader`. Connects to a ZMQ PUSH socket on the leader Pi
and receives joint positions as JSON.

- `config_so_network_leader.py` — Config: `remote_ip`, `port_zmq`, timeouts
- `so_network_leader.py` — `SONetworkLeader(Teleoperator)` implementation

### `bi_so_network_leader/`
Bimanual wrapper — composes two `SONetworkLeader` instances (left + right arms
on separate ZMQ ports). Drop-in replacement for `BiSOLeader`.

- `config_bi_so_network_leader.py` — Config with `left_arm_config` + `right_arm_config`
- `bi_so_network_leader.py` — `BiSONetworkLeader(Teleoperator)` implementation

## Deployment

These modules are **symlinked** into the lerobot teleoperators directory on the Pi
by `installation/05-install-network-teleop.sh`. This keeps the lerobot repo clean
while making the modules importable as `lerobot.teleoperators.so_network_leader`.

## Usage (on follower Pi)

```bash
# Single arm
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.id=bmh_follower_left \
    --teleop.type=so_network_leader \
    --teleop.remote_ip=192.168.0.200 \
    --teleop.port_zmq=5555

# Bimanual
# --robot.id=bmh_follower makes bi_so_follower look for calibration files
# bmh_follower_left.json and bmh_follower_right.json (written by 03-calibrate.sh)
lerobot-teleoperate \
    --robot.type=bi_so_follower \
    --robot.id=so_follower \
    --robot.left_arm_config.port=/dev/ttyACM0 \
    --robot.right_arm_config.port=/dev/ttyACM1 \
    --teleop.type=bi_so_network_leader \
    --teleop.left_arm_config.remote_ip=192.168.0.200 \
    --teleop.left_arm_config.port_zmq=5555 \
    --teleop.right_arm_config.remote_ip=192.168.0.200 \
    --teleop.right_arm_config.port_zmq=5557


lerobot-record \
    --robot.type=bi_so_follower \
    --robot.id=so_follower \
    --robot.left_arm_config.port=/dev/ttyACM0 \
    --robot.right_arm_config.port=/dev/ttyACM1 \
    --robot.left_arm_config.cameras='{"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}, "left_wrist": {"type": "opencv", "index_or_path": 2, "width": 320, "height": 240, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}}' \
    --robot.right_arm_config.cameras='{"right_wrist": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}}' \
    --teleop.type=bi_so_network_leader \
    --teleop.left_arm_config.remote_ip=192.168.0.200 \
    --teleop.left_arm_config.port_zmq=5555 \
    --teleop.right_arm_config.remote_ip=192.168.0.200 \
    --teleop.right_arm_config.port_zmq=5557 \
    --dataset.repo_id=my_local_sets/fun_set_2 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the Controller" \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --play_sounds=false

```

