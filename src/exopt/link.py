from typing import cast
import time

import zmq.utils.monitor

import numpy as np
import zmq

from exopt.trials import columns, gait_estimate, stop_reason

SAMPLE_PORT = 5555  # controller PUB -> driver SUB, one JSON dict per control step
PROFILE_PORT = 5556  # driver PUB -> controller SUB, one JSON dict per profile request or release
WARMUP = 100  # samples dropped after the ramp up
STALL = 10.0  # [s] without samples: the controller is gone


def controller_link() -> tuple[zmq.Socket, zmq.Socket]:
    """Controller side: bind a PUB for samples out and a SUB for profile updates in."""
    ctx = zmq.Context.instance()
    samples, profiles = ctx.socket(zmq.PUB), ctx.socket(zmq.SUB)
    # heartbeats drop a driver that vanished without closing (wifi, a dead tunnel), so it stops counting as a peer
    for socket in (samples, profiles):
        socket.setsockopt(zmq.HEARTBEAT_IVL, 1000)
        socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 3000)
    samples.bind(f"tcp://*:{SAMPLE_PORT}")
    profiles.bind(f"tcp://*:{PROFILE_PORT}")
    profiles.setsockopt_string(zmq.SUBSCRIBE, "")
    return samples, profiles


class Peers:
    """Connections open to bound sockets (dashboards, bo-loops), counted from their zmq monitors."""

    def __init__(self, *sockets: zmq.Socket):
        self.monitors = [s.get_monitor_socket(zmq.EVENT_ACCEPTED | zmq.EVENT_DISCONNECTED) for s in sockets]
        self.count = 0

    def __call__(self) -> int:
        for monitor in self.monitors:
            while monitor.poll(0):
                event = zmq.utils.monitor.recv_monitor_message(monitor)["event"]
                self.count += 1 if event == zmq.EVENT_ACCEPTED else -1
        return self.count


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


def collect_segment(samples: zmq.Socket, profiles: zmq.Socket, payload: dict) -> tuple[list[dict], str]:
    """Publish the profile and gather samples until payload["stop_rule"] stops it (trials.stop_reason), and why.

    WARMUP counts from the end of the controller's ramp up; valid gaits (trials.gait_slices) count after it. The segment
    returned ends with a gait. Raises ConnectionError after STALL seconds without samples (the controller streams from
    startup).
    """
    collected, phase = [], None
    rule = payload["stop_rule"]
    published_at, heard_at = None, time.monotonic()
    while True:
        # republish every second until the profile is fully ramped up (slow-joiner losses)
        if not collected and (published_at is None or time.monotonic() - published_at > 1.0):
            profiles.send_json(payload)
            published_at = time.monotonic()
        if not samples.poll(100):
            if time.monotonic() - heard_at > STALL:
                raise ConnectionError(f"no samples from the controller for {STALL:g} s")
            continue

        # keep only samples of the requested profile, once ramped up to full gain
        sample = cast(dict, samples.recv_json())
        heard_at = time.monotonic()
        if sample["profile_id"] != payload["id"] or sample["gain"] < 1.0:
            continue
        collected.append(sample)
        if len(collected) <= WARMUP:
            continue

        # a gait ends where leg 0's phase wraps: check the rule on the segment up to it
        wrapped = None not in (phase, sample["phase_0"]) and sample["phase_0"] - phase < -np.pi
        phase = sample["phase_0"]
        if wrapped:
            segment = collected[WARMUP:-1]
            if reason := stop_reason(*gait_estimate(columns(segment)), rule):
                return segment, reason
