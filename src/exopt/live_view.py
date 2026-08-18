"""Zero the encoders and stream leg state to a live position/velocity dashboard."""

import argparse
import sys
import time

WINDOW = 30.0  # [s] dashboard history window
SAMPLING_STEP = 0.01  # [s] streaming period


def stream():
    """Pi side: zero encoders at the current pose and publish position/velocity."""
    import pyCandle
    from exopt import zmq_link

    samples, _ = zmq_link.controller_link()

    input("Subject standing still: press enter to zero the position")
    candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True)
    ids = candle.ping()
    if len(ids) < 2:
        sys.exit(f"expected 2 motors, found {len(ids)}")
    for id in ids[:2]:
        candle.addMd80(id)
    for id in ids[:2]:
        if not candle.controlMd80SetEncoderZero(id):
            sys.exit(f"failed to zero encoder on drive {id}")
    candle.begin()

    print("Streaming... Ctrl-C to stop")
    start = time.monotonic()
    try:
        while True:
            samples.send_json(
                dict(
                    time=time.monotonic() - start,
                    position_0=candle.md80s[0].getPosition(),
                    position_1=candle.md80s[1].getPosition(),
                    velocity_0=candle.md80s[0].getVelocity(),
                    velocity_1=candle.md80s[1].getVelocity(),
                )
            )
            time.sleep(SAMPLING_STEP)
    finally:
        candle.end()


def dashboard(host: str, port: int):
    """Mac side: live plots of the last 30 s of position and velocity."""
    import threading
    from collections import deque
    from typing import cast

    import plotly.graph_objects as go
    from dash import Dash, Input, Output, dcc, html
    from exopt import zmq_link

    samples, _ = zmq_link.driver_link(host)
    buffer = deque(maxlen=int(2 * WINDOW / SAMPLING_STEP))

    # drain the SUB socket into the buffer from a background thread
    def pump():
        while True:
            buffer.append(cast(dict, samples.recv_json()))

    threading.Thread(target=pump, daemon=True).start()

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="position"),
            dcc.Graph(id="velocity"),
            dcc.Interval(id="tick", interval=200),
        ]
    )

    @app.callback(
        Output("position", "figure"), Output("velocity", "figure"), Input("tick", "n_intervals")
    )
    def refresh(_):
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
        return figures

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
