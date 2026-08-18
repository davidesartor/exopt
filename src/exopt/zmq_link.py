"""Driver <-> controller exchange: ZMQ pub/sub link and payload encoding."""

import jax.numpy as jnp
import zmq

from typing import cast
from exopt import rkhs_functions

# link parameters
SAMPLE_PORT = 5555  # controller PUB -> driver SUB, one JSON dict per control step
PROFILE_PORT = 5556  # driver PUB -> controller SUB, one JSON dict per profile swap


def controller_link(
    sample_port: int = SAMPLE_PORT, profile_port: int = PROFILE_PORT
) -> tuple[zmq.Socket, zmq.Socket]:
    """Controller side: bind a PUB for samples out and a SUB for profile updates in."""
    ctx = zmq.Context.instance()
    samples = ctx.socket(zmq.PUB)
    samples.bind(f"tcp://*:{sample_port}")
    profiles = ctx.socket(zmq.SUB)
    profiles.bind(f"tcp://*:{profile_port}")
    profiles.setsockopt_string(zmq.SUBSCRIBE, "")
    return samples, profiles


def driver_link(
    host: str = "localhost",
    sample_port: int = SAMPLE_PORT,
    profile_port: int = PROFILE_PORT,
) -> tuple[zmq.Socket, zmq.Socket]:
    """BO side: connect a SUB for samples in and a PUB for profiles out.

    With a remote host, reach it through an ssh tunnel:
    ssh -N -L 5555:localhost:5555 -L 5556:localhost:5556 user@host
    """
    ctx = zmq.Context.instance()
    samples = ctx.socket(zmq.SUB)
    samples.connect(f"tcp://{host}:{sample_port}")
    samples.setsockopt_string(zmq.SUBSCRIBE, "")
    profiles = ctx.socket(zmq.PUB)
    profiles.connect(f"tcp://{host}:{profile_port}")
    return samples, profiles


def latest_torque_profile(profiles: zmq.Socket) -> dict | None:
    """Drain the profile socket, returning the newest update if any."""
    newest = None
    while profiles.poll(0):
        newest = cast(dict, profiles.recv_json())
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

    Stops when the standard error of the mean penalized power falls below
    tol * max(|mean|, 1), bounded by min_samples and max_samples.
    """
    from exopt import acquisition  # driver-side only; keeps vlse off the controller

    profiles.send_json(payload)
    collected = []
    mean, m2 = 0.0, 0.0
    while True:
        # republish until the controller confirms the swap (slow-joiner losses)
        if not samples.poll(1000):
            profiles.send_json(payload)
            continue

        # keep only samples tagged with the requested profile
        sample = cast(dict, samples.recv_json())
        if sample["profile_id"] != payload["id"]:
            continue
        collected.append(sample)
        n = len(collected) - warmup
        if n < 1:
            continue

        # Welford update of the running objective estimate
        x = acquisition.sample_power(sample)
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


def profile_payload(profile: rkhs_functions.Profile) -> dict:
    """JSON-safe encoding of a candidate's Fourier coefficients."""
    return dict(
        sin=[float(c) for c in profile.sin],
        cos=[float(c) for c in profile.cos],
    )


def config_torque_profile(payload: dict) -> rkhs_functions.Profile:
    """Decode a payload back into a callable Profile."""
    return rkhs_functions.Profile(
        jnp.asarray(payload["sin"]), jnp.asarray(payload["cos"])
    )
