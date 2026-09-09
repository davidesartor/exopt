"""Zero the encoders and stream leg state to a live position/velocity dashboard."""

import argparse
import sys
import threading
import time

from exopt import MAX_TORQUE

WINDOW = 30.0  # [s] dashboard history window
SAMPLING_STEP = 0.01  # [s] streaming period
DRIVE_STATES = ["off", "zero"]  # unpowered, or enabled and commanded zero torque


def set_drives(candle, ids, enabled: bool) -> None:
    """Enable or disable both drives; config commands need the update loop stopped."""
    import pyCandle

    candle.end()
    for id, md in zip(ids, candle.md80s):
        if enabled:
            if not candle.controlMd80Mode(id, pyCandle.RAW_TORQUE):
                print(f"warning: failed to set torque mode on drive {id}")
            md.setMaxTorque(MAX_TORQUE)
            md.setTargetTorque(0.0)
        if not candle.controlMd80Enable(id, enabled):
            action = "enable" if enabled else "disable"
            print(f"warning: failed to {action} drive {id}")
    candle.begin()


def stream():
    """Pi side: zero encoders, publish leg state, and follow the dashboard drive toggle."""
    import pyCandle
    from exopt import zmq_link

    samples, commands = zmq_link.controller_link()

    input("Subject standing still: press enter to zero the position")
    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
    ids = candle.ping()[:2]
    if len(ids) < 2:
        sys.exit(f"expected 2 motors, found {len(ids)}")
    for id in ids:
        candle.addMd80(id)
    for id in ids:
        if not candle.controlMd80SetEncoderZero(id):
            sys.exit(f"failed to zero encoder on drive {id}")
    candle.begin()

    print("Streaming with the drives off... Ctrl-C to stop")
    start = time.monotonic()
    drives = "off"
    try:
        while True:
            # the dashboard repeats its desired state, so act only on a change
            command = zmq_link.latest_message(commands)
            if command is not None and command.get("drives") != drives:
                drives = command["drives"]
                set_drives(candle, ids, drives == "zero")
                print(f"drives: {drives}")

            # re-command zero every step so an enabled drive never holds a stale torque
            if drives == "zero":
                for md in candle.md80s:
                    md.setTargetTorque(0.0)

            samples.send_json(
                dict(
                    time=time.monotonic() - start,
                    drives=drives,
                    position_0=candle.md80s[0].getPosition(),
                    position_1=candle.md80s[1].getPosition(),
                    velocity_0=candle.md80s[0].getVelocity(),
                    velocity_1=candle.md80s[1].getVelocity(),
                )
            )
            time.sleep(SAMPLING_STEP)
    finally:
        for md in candle.md80s:
            md.setTargetTorque(0.0)
        candle.end()
        for id in ids:
            candle.controlMd80Enable(id, False)


def dashboard(host: str, port: int):
    """Mac side: live plots of the last 30 s of position and velocity, plus a drive toggle."""
    from collections import deque
    from typing import cast

    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dcc, html
    from exopt import zmq_link

    samples, commands = zmq_link.driver_link(host)
    buffer = deque(maxlen=int(2 * WINDOW / SAMPLING_STEP))
    command_lock = threading.Lock()

    # drain the SUB socket into the buffer from a background thread
    def pump():
        while True:
            buffer.append(cast(dict, samples.recv_json()))

    threading.Thread(target=pump, daemon=True).start()

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label("drives", style=dict(marginRight="1em")),
                    dcc.RadioItems(
                        id="drives",
                        options=[
                            dict(label="off (unpowered)", value="off"),
                            dict(label="zero torque", value="zero"),
                        ],
                        value="off",
                        inline=True,
                    ),
                    html.Span(id="status", style=dict(marginLeft="2em")),
                ],
                style=dict(display="flex", alignItems="center", padding="0.5em 1em"),
            ),
            dcc.Graph(id="position"),
            dcc.Graph(id="velocity"),
            dcc.Interval(id="tick", interval=200),
        ]
    )

    @app.callback(
        Output("position", "figure"),
        Output("velocity", "figure"),
        Output("status", "children"),
        Input("tick", "n_intervals"),
        State("drives", "value"),
    )
    def refresh(_, drives):
        # republish the desired state every tick: PUB/SUB drops messages sent before a join
        with command_lock:
            commands.send_json(dict(drives=drives))

        data = list(buffer)
        now = data[-1]["time"] if data else 0.0
        data = [s for s in data if s["time"] > now - WINDOW]
        t = [s["time"] for s in data]
        figures = []
        for kind, unit in [("position", "rad"), ("velocity", "rad/s")]:
            fig = go.Figure()
            for leg in (0, 1):
                fig.add_scatter(x=t, y=[s[f"{kind}_{leg}"] for s in data], name=f"leg {leg}")
            fig.update_layout(
                title=kind, yaxis_title=unit, xaxis_title="time [s]",
                margin=dict(l=40, r=10, t=40, b=40), height=350, uirevision="keep",
            )
            figures.append(fig)
        reported = data[-1].get("drives", "unknown") if data else "no samples"
        return *figures, f"controller reports: {reported}"

    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stream", "dashboard"])
    parser.add_argument("--host", default="localhost", help="controller host for dashboard mode")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    args = parser.parse_args()
    stream() if args.mode == "stream" else dashboard(args.host, args.port)


if __name__ == "__main__":
    main()
