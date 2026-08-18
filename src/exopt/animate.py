"""Side-view animation of the exo gait: fixed hip pivot, two swinging legs."""

import argparse
import math
import os
import webbrowser

import numpy as np
import plotly.graph_objects as go

from exopt import experiment_io

LEG_LENGTH = 1.0
COLORS = ["#d62728", "#1f77b4"]


def leg_endpoint(angle: float) -> tuple[float, float]:
    """Angle 0 is hanging straight down; positive swings forward (+x)."""
    return LEG_LENGTH * math.sin(angle), -LEG_LENGTH * math.cos(angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", default="experiments/mock-local")
    parser.add_argument("--swing", type=float, default=45.0, help="max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="playback frame rate")
    args = parser.parse_args()

    runs = sorted(experiment_io.run_numbers(args.exp_dir))

    # resample every run onto a uniform real-time grid
    segments = []
    for run in runs:
        trace = np.genfromtxt(f"{args.exp_dir}/run_{run}.txt", names=True)
        times = np.arange(trace["time"][0], trace["time"][-1], 1.0 / args.fps)
        resampled = {c: np.interp(times, trace["time"], trace[c]) for c in trace.dtype.names}
        loss = experiment_io.read_result(args.exp_dir, run)
        segments.append((run, times, resampled, loss))

    # one global angle scale so swings are comparable across runs
    peak = max(np.abs(np.stack([s["position_0"], s["position_1"]])).max()
               for _, _, s, _ in segments)
    scale = math.radians(args.swing) / peak

    def frame_data(seg, i):
        legs = []
        for leg in (0, 1):
            x, y = leg_endpoint(scale * seg[f"position_{leg}"][i])
            legs.append(
                go.Scatter(
                    x=[0, x], y=[0, y],
                    mode="lines+markers",
                    line=dict(color=COLORS[leg], width=6),
                    marker=dict(size=[14, 8]),
                    name=f"leg {leg} (torque {seg[f'torque_{leg}'][i]:+.2f} Nm)",
                )
            )
        return legs

    def banner(run, loss):
        return [dict(text=f"run {run} — loss {loss:.2f}", x=0.5, y=1.05,
                     xref="paper", yref="paper", showarrow=False, font=dict(size=18))]

    frames, run_starts = [], {}
    for run, times, seg, loss in segments:
        run_starts[run] = f"{run}:0"
        frames += [
            go.Frame(data=frame_data(seg, i), name=f"{run}:{i}",
                     layout=go.Layout(annotations=banner(run, loss)))
            for i in range(len(times))
        ]

    fig = go.Figure(data=frame_data(segments[0][2], 0), frames=frames)
    dt_ms = 1000.0 / args.fps
    fig.update_layout(
        annotations=banner(segments[0][0], segments[0][3]),
        xaxis=dict(range=[-1.2, 1.2], visible=False),
        yaxis=dict(range=[-1.3, 0.3], visible=False, scaleanchor="x"),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=dt_ms, redraw=True),
                                         transition=dict(duration=0))],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="run "),
                steps=[
                    dict(label=str(run), method="animate",
                         args=[[run_starts[run]], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                    for run in runs
                ],
            )
        ],
    )

    out = f"{args.exp_dir}/experiment_view.html"
    fig.write_html(out, auto_play=False)
    print(f"Wrote {out}: {len(runs)} runs, {len(frames)} frames")
    webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
