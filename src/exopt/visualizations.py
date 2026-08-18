"""Visualizations of an experiment folder: BO progress dashboard and gait animation."""

import argparse
import math
import os
import webbrowser
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from exopt import experiment_io, zmq_link

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
        (i, zmq_link.config_torque_profile(experiment_io.read_config(exp_dir, i)),
         experiment_io.read_result(exp_dir, i))
        for i in completed
    ]


def progress_figure(exp_dir: str, minimize: bool = True) -> go.Figure:
    """Dashboard: loss history with best-so-far, and every torque profile tried."""
    history = load_history(exp_dir)
    if not history:
        return go.Figure(layout=dict(title=f"{exp_dir} — no completed runs"))
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


def traces_figure(exp_dir: str, minimize: bool = True,
                  selected: list[int] | None = None) -> go.Figure:
    """Run diagnostics: position/velocity phase portraits and running mean mechanical power."""
    runs = sorted(experiment_io.run_numbers(exp_dir))
    if selected:
        runs = [r for r in runs if r in selected]
    if not runs:
        return go.Figure(layout=dict(title=f"{exp_dir} — no runs"))
    completed = [r for r in runs if r in experiment_io.config_numbers(exp_dir)]
    losses = {r: experiment_io.read_result(exp_dir, r) for r in completed}
    best_run = (min if minimize else max)(losses, key=losses.get) if losses else None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("phase portrait", "running mean mechanical power"),
    )

    # overlay every run, shaded old (light) to new (dark), best in red
    for k, run in enumerate(runs):
        trace = np.genfromtxt(f"{exp_dir}/run_{run}.txt", names=True)
        is_best = run == best_run
        shade = 0.2 + 0.8 * (k + 1) / len(runs)
        loss = losses.get(run)
        label = f"run {run}" + (f" (loss {loss:.3f})" if loss is not None else "")

        # left: velocity vs position, both legs
        for leg in (0, 1):
            fig.add_trace(
                go.Scatter(
                    x=trace[f"position_{leg}"], y=trace[f"velocity_{leg}"],
                    mode="lines", name=label, showlegend=False,
                    line=dict(
                        color=BEST_COLOR if is_best else f"rgba(31,119,180,{shade:.2f})",
                        width=2.5 if is_best else 1,
                    ),
                ),
                row=1, col=1,
            )

        # right: running mean of instantaneous power, both legs summed
        time = trace["time"] - trace["time"][0]
        power = sum(trace[f"torque_{leg}"] * trace[f"velocity_{leg}"] for leg in (0, 1))
        running_mean = np.cumsum(power) / np.arange(1, len(power) + 1)
        fig.add_trace(
            go.Scatter(
                x=time, y=running_mean, mode="lines", name=label, showlegend=False,
                line=dict(
                    color=BEST_COLOR if is_best else f"rgba(31,119,180,{shade:.2f})",
                    width=2.5 if is_best else 1.5,
                ),
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="position (rad)", row=1, col=1)
    fig.update_yaxes(title_text="velocity (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="power (W)", row=1, col=2)
    fig.update_layout(
        title=f"{exp_dir} — {len(runs)} runs"
              + (f", best run {best_run} (red)" if best_run is not None else ""),
    )
    return fig


def run_figure(exp_dir: str, run: int) -> go.Figure:
    """Single-run detail: phase portrait, power, and raw position/velocity/torque traces."""
    trace = np.genfromtxt(f"{exp_dir}/run_{run}.txt", names=True)
    time = trace["time"] - trace["time"][0]
    power = sum(trace[f"torque_{leg}"] * trace[f"velocity_{leg}"] for leg in (0, 1))
    running_mean = np.cumsum(power) / np.arange(1, len(power) + 1)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("phase portrait", "mechanical power",
                        "position", "torque"),
    )
    for leg in (0, 1):
        style = dict(line=dict(color=COLORS[leg]), legendgroup=f"leg {leg}",
                     name=f"leg {leg}")
        fig.add_trace(
            go.Scatter(x=trace[f"position_{leg}"], y=trace[f"velocity_{leg}"],
                       mode="lines", **style),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=trace[f"position_{leg}"], mode="lines",
                       showlegend=False, **style),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time, y=trace[f"torque_{leg}"], mode="lines",
                       showlegend=False, **style),
            row=2, col=2,
        )
    fig.add_trace(
        go.Scatter(x=time, y=power, mode="lines", name="instantaneous",
                   line=dict(color="rgba(31,119,180,0.4)", width=1)),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=time, y=running_mean, mode="lines", name="running mean",
                   line=dict(color=BEST_COLOR, width=2.5)),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="position (rad)", row=1, col=1)
    fig.update_yaxes(title_text="velocity (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="power (W)", row=1, col=2)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="position (rad)", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=2)
    fig.update_yaxes(title_text="torque (Nm)", row=2, col=2)

    loss = (experiment_io.read_result(exp_dir, run)
            if run in experiment_io.config_numbers(exp_dir) else None)
    fig.update_layout(
        title=f"{exp_dir} — run {run}"
              + (f", loss {loss:.4f}" if loss is not None else ""),
        height=700,
    )
    return fig


def run_dashboard(exp_dir: str, minimize: bool = True, port: int = 8050):
    """Live dashboard: run selector, refreshes progress and trace figures every 2 s."""
    from dash import Dash, Input, Output, State, dcc, html

    app = Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Tabs(
                id="view", value="experiment",
                children=[
                    dcc.Tab(label="experiment", value="experiment", children=[
                        dcc.Graph(id="progress"),
                        dcc.Graph(id="traces"),
                    ]),
                    dcc.Tab(label="run", value="run", children=[
                        dcc.Dropdown(id="run", clearable=False,
                                     style={"width": "200px"}),
                        dcc.Graph(id="detail"),
                    ]),
                ],
            ),
            dcc.Interval(id="tick", interval=2000),
        ]
    )

    @app.callback(
        [Output("run", "options"), Output("run", "value")],
        Input("tick", "n_intervals"),
        State("run", "value"),
    )
    def refresh_runs(_, current):
        runs = sorted(experiment_io.run_numbers(exp_dir))
        return runs, current if current in runs else (runs[-1] if runs else None)

    @app.callback(
        [Output("progress", "figure"), Output("traces", "figure")],
        Input("tick", "n_intervals"),
    )
    def refresh_experiment(_):
        return (progress_figure(exp_dir, minimize=minimize),
                traces_figure(exp_dir, minimize=minimize))

    @app.callback(
        Output("detail", "figure"),
        [Input("run", "value"), Input("tick", "n_intervals")],
    )
    def refresh_detail(run, _):
        return run_figure(exp_dir, run) if run is not None else go.Figure()

    app.run(debug=False, port=port)


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
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock-local",
                        help="experiment folder")
    parser.add_argument("--view", choices=["dashboard", "progress", "traces",
                                           "animation", "all"],
                        default="dashboard")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--minimize", action="store_true",
                        help="best-so-far tracks the minimum (default: maximize)")
    parser.add_argument("--swing", type=float, default=45.0, help="max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="playback frame rate")
    args = parser.parse_args()

    if args.view == "dashboard":
        webbrowser.open(f"http://127.0.0.1:{args.port}")
        run_dashboard(args.exp_dir, minimize=args.minimize, port=args.port)
        return

    views = ["progress", "traces", "animation"] if args.view == "all" else [args.view]
    for view in views:
        if view == "progress":
            fig = progress_figure(args.exp_dir, minimize=args.minimize)
        elif view == "traces":
            fig = traces_figure(args.exp_dir, minimize=args.minimize)
        else:
            fig = animation_figure(args.exp_dir, args.swing, args.fps)
        out = f"{args.exp_dir}/{view}.html"
        fig.write_html(out, auto_play=False)
        print(f"Wrote {out}")
        webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
