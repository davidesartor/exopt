"""exo-dashboard: what the exo does now, how the optimization goes, any trial in detail."""

from typing import NamedTuple, cast
from collections import deque
from functools import partial
import numpy as np
import plotly.graph_objects as go

import argparse
import math
import os
import threading
import time
import webbrowser

from exopt import MAX_TORQUE, link, trials
from exopt.profiles import EMBEDDINGS, Profile, atom

LIVE_WINDOW = 15.0  # [s] of leg traces shown
TORQUE_WINDOW = 4.0  # [s] of live torque overlaid on the applied profile
LIVE_STRIDE = 3  # plot every 3rd sample of the traces, so a refresh stays cheap
REFRESH_MS = 100  # live band refresh period (10 Hz)
STALE_AFTER = 2.0  # [s] without samples before the stream counts as lost


# the browser's setting, for the figures
THEME_JS = """
function(_, clicks, current) {
    const root = document.documentElement;
    const os = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const triggered = dash_clientside.callback_context.triggered.map((t) => t.prop_id);
    if (triggered.includes("theme-toggle.n_clicks")) {
        root.dataset.theme = (root.dataset.theme || os) === "dark" ? "light" : "dark";
    }
    const theme = root.dataset.theme || os;
    return theme === current ? dash_clientside.no_update : theme;
}
"""
CLICKABLE = ("loss", "surrogate", "profiles", "map")


def status_tiles(window: list[dict], s: dict, live: bool, current: dict | None, segment: dict,
                 history: list[trials.Trial], host: str) -> list[tuple]:
    """(label, value, detail, alert) for each status tile."""
    last = window[-1] if live and window else None
    period = gait_period(s) if s else None
    tiles = [
        ("controller", last["stage"], f"profile {last['profile_id'] or '—'}", False) if last
        else ("controller", "no stream", f"from {host}", True),
        ("gait period", f"{period:.2f} s", f"{60 / period:.0f} cycles/min", False) if period
        else ("gait period", "—", "waiting for the phase", False),
        ("gain", f"{100 * last['gain']:.0f}%", "full scale" if last["gain"] >= 1 else "ramping", False)
        if last else ("gain", "—", "", False),
    ]

    # the segment being collected, or the trial it just became
    if current and history and current["id"].startswith(f"{history[-1].index}-"):
        tiles.append(("segment estimate", "recorded", f"trial {history[-1].index} saved · proposing next", False))
    elif len(segment.get("time", [])) > 1:
        _, loss, se = running_estimate(segment)
        rule = current.get("stop_rule") if current else None
        gap = f" · {(loss[-1] - rule['best_mean']) / np.hypot(se[-1], rule['best_sd']):.1f} sd above best" if rule else ""
        tiles.append(("segment estimate", f"{loss[-1]:.3f}", f"± {se[-1]:.3f} over {len(loss)} samples{gap}", False))
    else:
        tiles.append(("segment estimate", "—", "collecting" if current else "no profile running", False))

    if history:
        best = min(history, key=lambda t: t.loss)
        tiles.append(("trials", str(len(history)), f"latest #{history[-1].index}: {history[-1].loss:.3f}", False))
        tiles.append(("best loss", f"{best.loss:.3f}", f"trial {best.index}", False))
    return tiles


def run_dashboard(exp_dir: str, host: str, port: int, warmup: int, mode: str):
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

    # drain the controller's sample stream into a rolling buffer from a background thread
    stream, _ = link.driver_link(host)
    buffer: deque[dict] = deque(maxlen=int(4 * LIVE_WINDOW / 0.01))
    received = dict(at=-math.inf)

    def pump():
        while True:
            sample = cast(dict, stream.recv_json())
            # a controller restart resets its clock: drop the previous run
            if buffer and sample["time"] < buffer[-1]["time"]:
                buffer.clear()
            buffer.append(sample)
            received["at"] = time.monotonic()

    threading.Thread(target=pump, daemon=True).start()

    graph = lambda id: html.Div(className="card", children=dcc.Graph(id=id, config=dict(displaylogo=False)))
    grid = lambda *ids: html.Div(className="grid", children=[graph(id) for id in ids])
    app = Dash(__name__, title="exopt")
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    app.layout = html.Div(className="page", children=[
        html.Header([
            html.H1("exopt session"), html.Span(f"{os.path.abspath(exp_dir)} · controller {host}"),
            html.Button("toggle theme", id="theme-toggle"),
        ]),
        html.Div(id="tiles", className="tiles"),
        html.H2("now"), grid("now-profile", "segment"), grid("live"),
        html.H2("optimization"), grid(*CLICKABLE),
        html.H2("trial"),
        html.Div(className="controls", children=[
            html.Button("← previous", id="previous"),
            dcc.Dropdown(id="trial", clearable=False, style=dict(width="140px")),
            html.Button("next →", id="next"),
            dcc.Checklist(id="follow", options=[dict(label=" follow latest", value="on")], value=["on"]),
            html.Span(id="summary"),
        ]),
        grid("trial-profile", "trial-power", "trial-convergence", "trial-portrait"),
        dcc.Store(id="selected"),
        dcc.Store(id="theme"),
        dcc.Interval(id="fast", interval=REFRESH_MS),
        dcc.Interval(id="slow", interval=2500),
    ])

    app.clientside_callback(
        THEME_JS, Output("theme", "data"),
        Input("fast", "n_intervals"), Input("theme-toggle", "n_clicks"), State("theme", "data"),
    )

    @app.callback(
        Output("tiles", "children"), Output("now-profile", "figure"), Output("segment", "figure"),
        Output("live", "figure"), Input("fast", "n_intervals"), Input("theme", "data"),
    )
    def refresh_now(_, theme):
        p = THEMES.get(theme, LIGHT)
        data = list(buffer)
        window = [x for x in data if x["time"] > data[-1]["time"] - LIVE_WINDOW] if data else []
        s = columns(window)
        live = time.monotonic() - received["at"] < STALE_AFTER
        current = trials.load_current(exp_dir)
        history = trials.load(exp_dir)

        # samples of the profile being collected, once at full gain, after the warmup
        segment = columns([x for x in data if current and x["profile_id"] == current["id"] and x["gain"] >= 1][warmup:])
        recent = columns([x for x in window if x["time"] > window[-1]["time"] - TORQUE_WINDOW]) if window else None

        tiles = [
            html.Div(className="tile alert" if alert else "tile", children=[
                html.Div(label, className="label"), html.Div(value, className="value"), html.Div(sub, className="sub"),
            ])
            for label, value, sub, alert in status_tiles(window, s, live, current, segment, history, host)
        ]
        profile = Profile.from_dict(current) if current else None
        return (
            tiles,
            profile_figure("applied profile vs live torque", profile, recent, "torque", p=p),
            estimate_figure("current segment: loss estimate", segment, current and current.get("stop_rule"), p=p),
            live_figure(s, p=p),
        )

    @app.callback(
        Output("selected", "data"), Output("trial", "value"), Output("trial", "options"), Output("follow", "value"),
        Input("slow", "n_intervals"), Input("trial", "value"), Input("previous", "n_clicks"), Input("next", "n_clicks"),
        *(Input(id, "clickData") for id in CLICKABLE), Input("follow", "value"),
        State("selected", "data"),
    )
    def select(_, dropdown, _previous, _next, *clicks_and_state):
        *clicks, follow, selected = clicks_and_state
        index = [t.index for t in trials.load(exp_dir)]
        latest = index[-1] if index else None
        trigger, choice = ctx.triggered_id, selected

        # a click on any trial marker or line, the dropdown, or the step buttons
        if trigger in CLICKABLE:
            data = ((clicks[CLICKABLE.index(trigger)] or {}).get("points") or [{}])[0].get("customdata")
            choice = int(data[0] if isinstance(data, list) else data) if data is not None else selected
        elif trigger == "trial":
            choice = dropdown
        elif trigger in ("previous", "next") and selected in index:
            step = 1 if trigger == "next" else -1
            choice = index[min(max(index.index(selected) + step, 0), len(index) - 1)]
        elif follow or selected not in index:
            choice = latest

        # picking a trial by hand stops following the latest one
        manual = trigger not in (None, "slow", "follow") and choice != latest
        options = [dict(label=f"trial {i}", value=i) for i in index]
        return no_update if choice == selected else choice, choice, options, [] if manual else follow

    @app.callback(
        *(Output(id, "figure") for id in CLICKABLE),
        Input("slow", "n_intervals"), Input("selected", "data"), Input("theme", "data"),
    )
    def refresh_optimization(_, selected, theme):
        history, p = trials.load(exp_dir), THEMES.get(theme, LIGHT)
        figures = (loss_figure, partial(surrogate_figure, mode=mode), profiles_figure, map_figure)
        return tuple(figure(history, selected, p=p) for figure in figures)

    @app.callback(
        Output("trial-profile", "figure"), Output("trial-power", "figure"), Output("trial-convergence", "figure"),
        Output("trial-portrait", "figure"), Output("summary", "children"),
        Input("selected", "data"), Input("slow", "n_intervals"), Input("theme", "data"),
    )
    def refresh_trial(selected, _, theme):
        trial = next((t for t in trials.load(exp_dir) if t.index == selected), None)
        p = THEMES.get(theme, LIGHT)
        if trial is None:
            return (placeholder(p, "no trial selected"),) * 4 + ("",)
        return (
            profile_figure("profile and its atoms", trial.profile, trial.samples, "measured_torque", atoms=True,
                           power_assist=trial.power_assist, p=p),
            power_figure(trial, p=p),
            estimate_figure("convergence of the estimate", trial.samples, trial.settings, p=p),
            portrait_figure(trial, p=p),
            trial_summary(trial),
        )

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock", help="experiment folder")
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--warmup", type=int, default=100, help="samples dropped after the ramp up (as bo-loop)")
    parser.add_argument("--mode", choices=EMBEDDINGS, default="functional", help="surrogate inputs (as bo-loop)")
    parser.add_argument("--animation", action="store_true", help="export the gait animation instead")
    parser.add_argument("--swing", type=float, default=45.0, help="animation max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="animation frame rate")
    args = parser.parse_args()

    remote, updated, warning = update_checkout()
    # new commits: start over on them, with their dependencies
    if updated:
        try:
            sys.exit(subprocess.call(["uv", "run", "--project", str(REPO), "--group", "dev", "start", *sys.argv[1:]]))
        except OSError as error:
            warning = f"pulled new commits but could not restart on them ({error}): restart the dashboard"
    if warning:
        print(warning)
    run_dashboard(args.exp_dir, args.host, args.port, (args.exo_host, args.exo_user, args.exo_port), args.atoms,
                  MOCK_CONTROLLER if args.mock else CONTROLLER, remote, warning)


if __name__ == "__main__":
    main()
