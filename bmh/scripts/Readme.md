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
    --fps 60

# Single arm only
python so_leader_host.py \
    --left_port /dev/ttyACM0 \
    --left_zmq_port 5555
```

## Wire Protocol

- **Transport**: ZMQ PUSH → PULL, `tcp://*:{port}`
- **Serialization**: JSON string
- **Message**: `{"shoulder_pan.pos": 45.2, "shoulder_lift.pos": -12.1, ...}` (6 floats)
- **Socket option**: `CONFLATE=1` (latest-only, no buffering)
- **Rate**: configurable, default 60 Hz