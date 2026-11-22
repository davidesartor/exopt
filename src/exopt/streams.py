from typing import cast

import zmq.utils.monitor

import zmq

SAMPLE_PORT = 5555  # controller PUB -> driver SUB, one JSON dict per control step
PROFILE_PORT = 5556  # driver PUB -> controller SUB, one JSON dict per profile request or release
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
    """Laptop side: connect a SUB for samples in and a PUB for profiles out (the Pi's ports are forwarded here, raspberrypi.Tunnel)."""
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
