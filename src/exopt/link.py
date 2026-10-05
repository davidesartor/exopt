"""Driver <-> controller ZMQ pub/sub link and segment collection."""

import time
from typing import cast

import zmq

from exopt.trials import penalized_power

SAMPLE_PORT = 5555  # controller PUB -> driver SUB, one JSON dict per control step
PROFILE_PORT = 5556  # driver PUB -> controller SUB, one JSON dict per profile swap


def controller_link() -> tuple[zmq.Socket, zmq.Socket]:
    """Controller side: bind a PUB for samples out and a SUB for profile updates in."""
    ctx = zmq.Context.instance()
    samples = ctx.socket(zmq.PUB)
    samples.bind(f"tcp://*:{SAMPLE_PORT}")
    profiles = ctx.socket(zmq.SUB)
    profiles.bind(f"tcp://*:{PROFILE_PORT}")
    profiles.setsockopt_string(zmq.SUBSCRIBE, "")
    return samples, profiles


def driver_link(host: str = "localhost") -> tuple[zmq.Socket, zmq.Socket]:
    """Mac side: connect a SUB for samples in and a PUB for profiles out.

    With a remote host, reach it through an ssh tunnel:
    ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@host
    """
    ctx = zmq.Context.instance()
    samples = ctx.socket(zmq.SUB)
    samples.connect(f"tcp://{host}:{SAMPLE_PORT}")
    samples.setsockopt_string(zmq.SUBSCRIBE, "")
    profiles = ctx.socket(zmq.PUB)
    profiles.connect(f"tcp://{host}:{PROFILE_PORT}")
    return samples, profiles


def latest_message(socket: zmq.Socket) -> dict | None:
    """Drain a SUB socket, returning the newest message if any."""
    newest = None
    while socket.poll(0):
        newest = cast(dict, socket.recv_json())
    return newest


def collect_segment(
    samples: zmq.Socket,
    profiles: zmq.Socket,
    payload: dict,
    warmup: int,
    tol: float = 0.05,
    min_samples: int = 100,
    max_samples: int = 2000,
) -> list[dict]:
    """Publish the profile and gather samples until the objective estimate converges.

    Warmup counts from the end of the controller's blend. Stops when the standard
    error of the mean penalized power falls below tol * max(|mean|, 1), bounded by
    min_samples and max_samples.
    """
    collected = []
    mean, m2 = 0.0, 0.0
    published_at = None
    while True:
        # republish every second until the controller confirms the swap (slow-joiner losses)
        if not collected and (published_at is None or time.monotonic() - published_at > 1.0):
            profiles.send_json(payload)
            published_at = time.monotonic()
        if not samples.poll(100):
            continue

        # keep only samples of the requested profile, once fully blended in
        sample = cast(dict, samples.recv_json())
        if sample["profile_id"] != payload["id"] or sample["blend"] < 1.0:
            continue
        collected.append(sample)
        n = len(collected) - warmup
        if n < 1:
            continue

        # Welford update of the running objective estimate
        x = float(penalized_power(sample))
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)

        # stop once the estimate is stable (or the budget is exhausted)
        if n >= max_samples:
            break
        if n >= min_samples:
            sem = (m2 / (n - 1) / n) ** 0.5
            if sem < tol * max(abs(mean), 1.0):
                break
    return collected[warmup:]
