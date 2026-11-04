"""exo-dashboard: live leg state from the controller, BO progress of a folder, gait animation."""

import argparse
import math
import os
import threading
import webbrowser
from collections import deque
from typing import cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from exopt import MAX_TORQUE, link, trials

LIVE_WINDOW = 30.0  # [s] live history shown
LIVE_STRIDE = 4  # plot every 4th sample, so a refresh stays well under its interval
LEG_LENGTH = 1.0  # leg segment length in animation units
COLORS = ["#d62728", "#1f77b4"]  # one color per leg
BEST_COLOR = "#d62728"
STAGE_RGB = ["31,119,180", "255,127,14", "44,160,44", "148,103,189"]  # indexed by atoms - 1


def stage_rgb(atoms: int) -> str:
    return STAGE_RGB[(atoms - 1) % len(STAGE_RGB)]


def visibility(history: list[trials.Trial]) -> np.ndarray:
    """Line opacity per trial from its loss rank: the best opaque, the worst faint."""
    rank = np.argsort(np.argsort([t.loss for t in history]))
    quality = 1.0 - rank / max(len(history) - 1, 1)
    return 0.05 + 0.95 * quality**2


def trial_color(trial: trials.Trial, alpha: float, stage_best: set[int]) -> str:
    """Stage color at the given opacity, red for the best trial of each stage."""
    return BEST_COLOR if trial.index in stage_best else f"rgba({stage_rgb(trial.profile.atoms)},{alpha:.2f})"


def stages(history: list[trials.Trial]) -> dict[int, list[trials.Trial]]:
    """Trials grouped by the number of atoms of their profile, in order of appearance."""
    grouped: dict[int, list[trials.Trial]] = {}
    for t in history:
        grouped.setdefault(t.profile.atoms, []).append(t)
    return grouped


def stage_name(atoms: int) -> str:
    return "1 atom" if atoms == 1 else f"{atoms} atoms"


def live_figure(window: list[dict]) -> go.Figure:
    """Position, velocity, commanded torque and estimated phase of both legs over the live window."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("position", "velocity", "torque", "phase"))
    t = [s["time"] for s in window]
    for row, (kind, unit) in enumerate([("position", "rad"), ("velocity", "rad/s"), ("torque", "Nm"), ("phase", "rad")], 1):
        for leg in (0, 1):
            fig.add_scatter(
                x=t, y=[s[f"{kind}_{leg}"] for s in window], row=row, col=1,
                name=f"leg {leg}", legendgroup=f"leg {leg}", showlegend=row == 1,
                line=dict(color=COLORS[leg]),
            )
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_layout(height=1000, margin=dict(l=40, r=10, t=40, b=40), uirevision="keep")
    return fig


def progress_figure(history: list[trials.Trial]) -> go.Figure:
    """Loss per trial with each stage's best so far, and every torque profile tried, better ones more opaque."""
    if not history:
        return go.Figure(layout=dict(title="no trials shown"))
    grouped = stages(history)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("loss per trial", "torque profiles over gait phase"))
    summary = []
    for atoms, group in grouped.items():
        rgb, name = stage_rgb(atoms), stage_name(atoms)
        index = [t.index for t in group]
        losses = np.array([t.loss for t in group])
        best = int(losses.argmin())
        summary.append(f"{name}: {len(group)} trials, best {losses[best]:.3f} (trial {index[best]})")

        # stage band, its trial losses and its own best-so-far staircase
        fig.add_vrect(
            x0=index[0] - 0.5, x1=index[-1] + 0.5, fillcolor=f"rgba({rgb},0.1)", line_width=0,
            annotation_text=name, annotation_position="top left", row=1, col=1,
        )
        fig.add_scatter(
            x=index, y=losses, mode="markers", name=name, legendgroup=name,
            marker=dict(size=9, color=f"rgb({rgb})"), row=1, col=1,
        )
        fig.add_scatter(
            x=index, y=np.minimum.accumulate(losses), mode="lines", name=f"{name} best so far", legendgroup=name,
            line=dict(color=f"rgb({rgb})", width=2, shape="hv"), row=1, col=1,
        )

    # every profile overlaid, worst first so the best are drawn on top
    phase = np.linspace(0.0, 2.0 * math.pi, 200)
    alpha = visibility(history)
    stage_best = {min(group, key=lambda t: t.loss).index for group in grouped.values()}
    for k in np.argsort(alpha):
        t = history[k]
        fig.add_scatter(
            x=phase / (2.0 * math.pi), y=MAX_TORQUE * np.asarray(t.profile(phase)),
            mode="lines", name=f"trial {t.index} (loss {t.loss:.3f})", showlegend=False,
            line=dict(color=trial_color(t, alpha[k], stage_best), width=3 if t.index in stage_best else 1.5),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="trial", row=1, col=1)
    fig.update_yaxes(title_text="loss", row=1, col=1)
    fig.update_xaxes(title_text="gait phase (cycles)", row=1, col=2)
    fig.update_yaxes(title_text="torque (Nm)", row=1, col=2)
    fig.update_layout(title=" · ".join(summary) + " (best of each stage red)")
    return fig


def traces_figure(history: list[trials.Trial]) -> go.Figure:
    """Phase portraits and running mean penalized power of every trial, better ones more opaque."""
    if not history:
        return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("phase portrait", "running mean penalized power"))
    alpha = visibility(history)
    stage_best = {min(group, key=lambda t: t.loss).index for group in stages(history).values()}
    for k in np.argsort(alpha):
        t, s = history[k], history[k].samples
        style = dict(
            mode="lines", name=f"trial {t.index} (loss {t.loss:.3f})", showlegend=False,
            line=dict(color=trial_color(t, alpha[k], stage_best), width=2.5 if t.index in stage_best else 1),
        )
        for leg in (0, 1):
            fig.add_scatter(x=s[f"position_{leg}"], y=s[f"velocity_{leg}"], row=1, col=1, **style)
        power = trials.penalized_power(s)
        fig.add_scatter(
            x=s["time"] - s["time"][0], y=np.cumsum(power) / np.arange(1, len(power) + 1),
            row=1, col=2, **style,
        )
    fig.update_xaxes(title_text="position (rad)", row=1, col=1)
    fig.update_yaxes(title_text="velocity (rad/s)", row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="power (W)", row=1, col=2)
    return fig


def trial_figure(trial: trials.Trial) -> go.Figure:
    """One trial in detail: phase portrait, power, position and torque traces."""
    s = trial.samples
    time = s["time"] - s["time"][0]
    power = trials.penalized_power(s)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("phase portrait", "penalized power", "position", "torque"))
    for leg in (0, 1):
        style = dict(mode="lines", line=dict(color=COLORS[leg]), legendgroup=f"leg {leg}", name=f"leg {leg}")
        fig.add_scatter(x=s[f"position_{leg}"], y=s[f"velocity_{leg}"], row=1, col=1, **style)
        fig.add_scatter(x=time, y=s[f"position_{leg}"], showlegend=False, row=2, col=1, **style)
        fig.add_scatter(x=time, y=s[f"torque_{leg}"], showlegend=False, row=2, col=2, **style)
        fig.add_scatter(
            x=time, y=s[f"measured_torque_{leg}"], showlegend=False, row=2, col=2,
            mode="lines", line=dict(color=COLORS[leg], dash="dot"),
        )
    fig.add_scatter(
        x=time, y=power, mode="lines", name="instantaneous",
        line=dict(color="rgba(31,119,180,0.4)", width=1), row=1, col=2,
    )
    fig.add_scatter(
        x=time, y=np.cumsum(power) / np.arange(1, len(power) + 1), mode="lines",
        name="running mean", line=dict(color=BEST_COLOR, width=2.5), row=1, col=2,
    )
    for row, col, x, y in [
        (1, 1, "position (rad)", "velocity (rad/s)"),
        (1, 2, "time (s)", "power (W)"),
        (2, 1, "time (s)", "position (rad)"),
        (2, 2, "time (s)", "torque (Nm), measured dotted"),
    ]:
        fig.update_xaxes(title_text=x, row=row, col=col)
        fig.update_yaxes(title_text=y, row=row, col=col)
    fig.update_layout(title=f"trial {trial.index}, loss {trial.loss:.4f}", height=700)
    return fig


def run_dashboard(exp_dir: str, host: str, port: int):
    """Dash app: live controller stream, experiment progress, and per-trial detail."""
    from dash import Dash, Input, Output, State, dcc, html

    # drain the controller's sample stream into a rolling buffer from a background thread
    samples, _ = link.driver_link(host)
    buffer = deque(maxlen=int(2 * LIVE_WINDOW / 0.01))

    def pump():
        while True:
            sample = cast(dict, samples.recv_json())
            # a controller restart resets its clock: drop the previous run
            if buffer and sample["time"] < buffer[-1]["time"]:
                buffer.clear()
            buffer.append(sample)

    threading.Thread(target=pump, daemon=True).start()

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Tabs(value="live", children=[
            dcc.Tab(label="live", value="live", children=[
                html.Div(id="status", style=dict(padding="0.5em 1em")),
                dcc.Graph(id="live"),
            ]),
            dcc.Tab(label="experiment", value="experiment", children=[
                dcc.Checklist(id="stages", inline=True, style=dict(padding="0.5em 1em")),
                dcc.Graph(id="progress"),
                dcc.Graph(id="traces"),
            ]),
            dcc.Tab(label="trial", value="trial", children=[
                dcc.Dropdown(id="trial", clearable=False, style=dict(width="200px")),
                dcc.Graph(id="detail"),
            ]),
        ]),
        dcc.Interval(id="fast", interval=500),
        dcc.Interval(id="slow", interval=2000),
    ])

    @app.callback(Output("live", "figure"), Output("status", "children"), Input("fast", "n_intervals"))
    def refresh_live(_):
        data = list(buffer)
        if not data:
            return go.Figure(), f"no samples from {host}"
        last = data[-1]
        window = [s for s in data if s["time"] > last["time"] - LIVE_WINDOW][::LIVE_STRIDE]
        status = f"{last['stage']} · profile {last['profile_id']} · blend {last['blend']:.2f}"
        return live_figure(window), status

    @app.callback(
        Output("progress", "figure"), Output("traces", "figure"),
        Output("trial", "options"), Output("trial", "value"),
        Output("stages", "options"), Output("stages", "value"),
        Input("slow", "n_intervals"), Input("stages", "value"),
        State("trial", "value"), State("stages", "options"),
    )
    def refresh_experiment(_, shown, selected, known):
        history = trials.load(exp_dir)
        index = [t.index for t in history]
        selected = selected if selected in index else (index[-1] if index else None)

        # stages start shown, including ones that appear mid-session
        options = [{"label": stage_name(h), "value": h} for h in stages(history)]
        known = {o["value"] for o in known or []}
        shown = (shown or []) + [o["value"] for o in options if o["value"] not in known]
        history = [t for t in history if t.profile.atoms in shown]
        return progress_figure(history), traces_figure(history), index, selected, options, shown

    @app.callback(Output("detail", "figure"), Input("trial", "value"), Input("slow", "n_intervals"))
    def refresh_detail(selected, _):
        trial = next((t for t in trials.load(exp_dir) if t.index == selected), None)
        return trial_figure(trial) if trial else go.Figure()

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


def animation_figure(history: list[trials.Trial], swing: float = 45.0, fps: float = 25.0) -> go.Figure:
    """Side-view animation of the gait: fixed hip pivot, two swinging legs, one segment per trial."""
    # resample every trial onto a uniform real-time grid
    segments = []
    for t in history:
        s = t.samples
        times = np.arange(s["time"][0], s["time"][-1], 1.0 / fps)
        columns = [f"{c}_{leg}" for c in ("position", "torque") for leg in (0, 1)]
        segments.append((t, {c: np.interp(times, s["time"], s[c]) for c in columns}))

    # one global angle scale so swings are comparable across trials
    peak = max(np.abs(np.stack([s["position_0"], s["position_1"]])).max() for _, s in segments)
    scale = math.radians(swing) / peak

    def frame_data(seg: dict, i: int) -> list[go.Scatter]:
        # angle 0 hangs straight down, positive swings forward (+x)
        legs = []
        for leg in (0, 1):
            angle = scale * seg[f"position_{leg}"][i]
            legs.append(go.Scatter(
                x=[0, LEG_LENGTH * math.sin(angle)], y=[0, -LEG_LENGTH * math.cos(angle)],
                mode="lines+markers", line=dict(color=COLORS[leg], width=6), marker=dict(size=[14, 8]),
                name=f"leg {leg} (torque {seg[f'torque_{leg}'][i]:+.2f} Nm)",
            ))
        return legs

    def banner(trial: trials.Trial) -> list[dict]:
        return [dict(text=f"trial {trial.index} — loss {trial.loss:.2f}", x=0.5, y=1.05,
                     xref="paper", yref="paper", showarrow=False, font=dict(size=18))]

    # flatten every trial into one frame sequence, the slider jumping to each trial's start
    frames = [
        go.Frame(data=frame_data(seg, i), name=f"{t.index}:{i}", layout=go.Layout(annotations=banner(t)))
        for t, seg in segments
        for i in range(len(seg["position_0"]))
    ]
    fig = go.Figure(data=frame_data(segments[0][1], 0), frames=frames)
    fig.update_layout(
        annotations=banner(segments[0][0]),
        xaxis=dict(range=[-1.2, 1.2], visible=False),
        yaxis=dict(range=[-1.3, 0.3], visible=False, scaleanchor="x"),
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="Play", method="animate",
            args=[None, dict(frame=dict(duration=1000.0 / fps, redraw=True), transition=dict(duration=0))],
        )])],
        sliders=[dict(currentvalue=dict(prefix="trial "), steps=[
            dict(label=str(t.index), method="animate",
                 args=[[f"{t.index}:0"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for t, _ in segments
        ])],
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock", help="experiment folder")
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--animation", action="store_true", help="export the gait animation instead")
    parser.add_argument("--swing", type=float, default=45.0, help="animation max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="animation frame rate")
    args = parser.parse_args()

    if not args.animation:
        run_dashboard(args.exp_dir, args.host, args.port)
        return
    out = f"{args.exp_dir}/animation.html"
    animation_figure(trials.load(args.exp_dir), args.swing, args.fps).write_html(out, auto_play=False)
    print(f"Wrote {out}")
    webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
