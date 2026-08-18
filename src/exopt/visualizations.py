"""Visualizations of an experiment folder: BO progress dashboard and gait animation."""

import argparse
import math
import os
import webbrowser
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from exopt import experiment_io, zmq_link
from exopt.rkhs_functions import Profile

# drawing parameters
LEG_LENGTH = 1.0  # leg segment length in plot units
COLORS = ["#d62728", "#1f77b4"]  # one color per leg
BEST_COLOR = "#d62728"


def load_history(exp_dir: str):
    """Completed runs as (run, profile, loss) triples, in run order."""
    completed = sorted(
        experiment_io.config_numbers(exp_dir) & experiment_io.run_numbers(exp_dir)
    )
    return [
        (i, decode_profile(experiment_io.read_config(exp_dir, i)),
         experiment_io.read_result(exp_dir, i))
        for i in completed
    ]


def decode_profile(payload: dict) -> Profile:
    """Decode a config into a Profile, accepting the legacy amplitude/phase format."""
    if "sin" in payload:
        return zmq_link.config_torque_profile(payload)
    a, p = payload["amplitude"], payload["phase"]
    return Profile(np.array([a * math.cos(p)]), np.array([a * math.sin(p)]))


def progress_figure(exp_dir: str, minimize: bool = True) -> go.Figure:
    """Dashboard: loss history with best-so-far, and every torque profile tried."""
    history = load_history(exp_dir)
    runs = [i for i, _, _ in history]
    losses = np.array([loss for _, _, loss in history])
    best = np.minimum.accumulate(losses) if minimize else np.maximum.accumulate(losses)
    best_run = runs[int(losses.argmin() if minimize else losses.argmax())]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("loss per run", "torque profiles over gait phase"),
    )

    # left: per-run losses and the running best
    fig.add_trace(
        go.Scatter(x=runs, y=losses, mode="markers", name="run loss",
                   marker=dict(size=9, color="#1f77b4")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=runs, y=best, mode="lines", name="best so far",
                   line=dict(color=BEST_COLOR, width=2, shape="hv")),
        row=1, col=1,
    )

    # right: each candidate profile over one gait cycle, shaded old (light) to new (dark)
    phase = np.linspace(0.0, 2.0 * math.pi, 200)
    for k, (run, profile, loss) in enumerate(history):
        is_best = run == best_run
        shade = 0.25 + 0.75 * (k + 1) / len(history)
        fig.add_trace(
            go.Scatter(
                x=phase / (2.0 * math.pi), y=np.asarray(profile(phase)),
                mode="lines", name=f"run {run} (loss {loss:.3f})",
                showlegend=False,
                line=dict(
                    color=BEST_COLOR if is_best else f"rgba(31,119,180,{shade:.2f})",
                    width=3 if is_best else 1.5,
                ),
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="run", row=1, col=1)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_xaxes(title_text="gait phase (cycles)", row=1, col=2)
    fig.update_yaxes(title_text="torque (Nm)", row=1, col=2)
    fig.update_layout(
        title=f"{exp_dir} — {len(runs)} runs, best {best[-1]:.4f} (run {best_run}, red)",
    )
    return fig


def leg_endpoint(angle: float) -> tuple[float, float]:
    """Angle 0 is hanging straight down; positive swings forward (+x)."""
    return LEG_LENGTH * math.sin(angle), -LEG_LENGTH * math.cos(angle)


def animation_figure(exp_dir: str, swing: float = 45.0, fps: float = 25.0) -> go.Figure:
    """Side-view animation of the exo gait: fixed hip pivot, two swinging legs."""
    runs = sorted(experiment_io.run_numbers(exp_dir))

    # resample every run onto a uniform real-time grid
    segments = []
    for run in runs:
        trace = np.genfromtxt(f"{exp_dir}/run_{run}.txt", names=True)
        times = np.arange(trace["time"][0], trace["time"][-1], 1.0 / fps)
        resampled = {c: np.interp(times, trace["time"], trace[c]) for c in trace.dtype.names}
        loss = experiment_io.read_result(exp_dir, run)
        segments.append((run, times, resampled, loss))

    # one global angle scale so swings are comparable across runs
    peak = max(np.abs(np.stack([s["position_0"], s["position_1"]])).max()
               for _, _, s, _ in segments)
    scale = math.radians(swing) / peak

    def frame_data(seg, i):
        # both legs as hip-to-foot segments at frame i
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

    # flatten every run into one frame sequence, remembering where each run starts
    frames, run_starts = [], {}
    for run, times, seg, loss in segments:
        run_starts[run] = f"{run}:0"
        frames += [
            go.Frame(data=frame_data(seg, i), name=f"{run}:{i}",
                     layout=go.Layout(annotations=banner(run, loss)))
            for i in range(len(times))
        ]

    # assemble the figure with a play button and a run-selection slider
    fig = go.Figure(data=frame_data(segments[0][2], 0), frames=frames)
    dt_ms = 1000.0 / fps
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
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", nargs="?", choices=["progress", "animation", "all"],
                        default="progress")
    parser.add_argument("--exp-dir", default="experiments/mock-local")
    parser.add_argument("--maximize", action="store_true",
                        help="best-so-far tracks the maximum instead of the minimum")
    parser.add_argument("--swing", type=float, default=45.0, help="max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="playback frame rate")
    args = parser.parse_args()

    views = ["progress", "animation"] if args.view == "all" else [args.view]
    for view in views:
        if view == "progress":
            fig = progress_figure(args.exp_dir, minimize=not args.maximize)
        else:
            fig = animation_figure(args.exp_dir, args.swing, args.fps)
        out = f"{args.exp_dir}/{view}.html"
        fig.write_html(out, auto_play=False)
        print(f"Wrote {out}")
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
