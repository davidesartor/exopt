from typing import cast
import time

import numpy as np
import zmq

from exopt.trials import gap_to_best, penalized_power

SAMPLE_PORT = 5555  # controller PUB -> driver SUB, one JSON dict per control step
PROFILE_PORT = 5556  # driver PUB -> controller SUB, one JSON dict per profile request or release


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
    # heartbeat so a controller restart behind the ssh tunnel drops and reconnects the link
    samples.setsockopt(zmq.HEARTBEAT_IVL, 1000)
    samples.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3000)
    samples.connect(f"tcp://{host}:{SAMPLE_PORT}")
    samples.setsockopt_string(zmq.SUBSCRIBE, "")
    profiles = ctx.socket(zmq.PUB)
    profiles.setsockopt(zmq.HEARTBEAT_IVL, 1000)
    profiles.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3000)
    profiles.connect(f"tcp://{host}:{PROFILE_PORT}")
    return samples, profiles


def latest_message(socket: zmq.Socket) -> dict | None:
    """Drain a SUB socket, returning the newest message if any."""
    newest = None
    while socket.poll(0):
        newest = cast(dict, socket.recv_json())
    return newest


def release(profiles: zmq.Socket) -> None:
    """Ask the controller to ramp the applied profile down to zero torque."""
    profiles.send_json(dict(id=None))


def collect_segment(
    samples: zmq.Socket,
    profiles: zmq.Socket,
    payload: dict,
    warmup: int,
    min_samples: int = 100,
    max_samples: int = 2000,
) -> list[dict]:
    """Publish the profile and gather samples until it is unlikely to beat the best.

    Warmup counts from the end of the controller's ramp up. With payload["stop_rule"]
    (best posterior mean, sd, z) the segment ends once its running loss sits more
    than z standard deviations above the best; otherwise it runs to max_samples.
    """
    collected, power = [], []
    rule = payload.get("stop_rule")
    published_at = None
    while True:
        # republish every second until the profile is fully ramped up (slow-joiner losses)
        if not collected and (published_at is None or time.monotonic() - published_at > 1.0):
            profiles.send_json(payload)
            published_at = time.monotonic()
        if not samples.poll(100):
            continue

        # keep only samples of the requested profile, once ramped up to full gain
        sample = cast(dict, samples.recv_json())
        if sample["profile_id"] != payload["id"] or sample["gain"] < 1.0:
            continue
        collected.append(sample)
        n = len(collected) - warmup
        if n < 1:
            continue

        # stop once the segment is confidently worse than the best (or the budget is exhausted)
        power.append(float(penalized_power(sample)))
        if n >= max_samples:
            break
        if rule and n >= min_samples:
            if gap_to_best(np.asarray(power), rule["best_mean"], rule["best_sd"]) > rule["z"]:
                break
    return collected[warmup:]
