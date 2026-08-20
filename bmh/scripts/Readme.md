# Leader Server — ZMQ Position Streamer

This directory contains `so_leader_host.py` — a standalone script that runs
on the **leader Pi** (Pi A).

It reads joint positions from one or two SO-101 leader arms via USB serial
and streams them over ZMQ PUSH sockets to the follower Pi.

## Usage

```bash
# Both arms (bimanual)
python so_leader_host.py \
    --left_port /dev/ttyACM0 \
    --right_port /dev/ttyACM1 \
    --left_zmq_port 5555 \
    --right_zmq_port 5557 \
    --left_with_head \
    --fps 60

# Single arm only
python so_leader_host.py \
    --left_port /dev/ttyACM0 \
    --left_zmq_port 5555
```

`--left_with_head` (default: enabled; disable with `--no-left_with_head`)
controls whether the left arm's bus also carries the BMH-101 head servos
(`head_pan`/`head_tilt`, IDs 8/9). The follower side must be configured with
the matching `with_head` value.

## Wire Protocol

- **Transport**: ZMQ PUSH → PULL, `tcp://*:{port}`
- **Serialization**: JSON string
- **Message**: `{"shoulder_pan.pos": 45.2, "shoulder_lift.pos": -12.1, ...}` (7 floats per arm; 9 on the left arm with head enabled — adds `head_pan.pos`/`head_tilt.pos`)
- **Socket option**: `CONFLATE=1` (latest-only, no buffering)
- **Rate**: configurable, default 60 Hz
