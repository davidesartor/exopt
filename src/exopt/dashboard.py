"""start: connect to the exo, run an experiment, review any experiment."""

from typing import NamedTuple, cast
from collections import deque
from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

import numpy as np
import plotly.graph_objects as go

from exopt import experiment, optimization_loop, raspberrypi, streams
from exopt.figures import (CLICKABLE, LIGHT, LIVE_HEIGHT, LIVE_ROWS, LIVE_STRIDE, LIVE_WINDOW, PAGE_CSS, THEMES, TRIAL, WEB, Palette,
                           break_wraps, cycles, estimate_figure, experiment_figure, gait_period, live_figure,
                           load_trials, loss_figure, placeholder, portrait_figure, power_figure, profile_figure,
                           profiles_figure, review_headline, trial_summary)

columns = experiment.columns
BUFFER_SECONDS = 60.0  # [s] of samples kept at full rate
TORQUE_WINDOW = 4.0  # [s] of live torque overlaid on the applied profile
CONTROLLER, MOCK, DIAGNOSE = "src/exopt/control_loop.py", "src/exopt/drives_mock.py", "src/exopt/diagnose.py"
REFRESH_MS = 1000  # live band refresh period (1 Hz)
FEED_MS = 1000 / 60  # live traces and segment estimate pulled and drawn at 60 Hz
FEED_POINTS = int((LIVE_WINDOW + 1.0) / 0.01 / LIVE_STRIDE)  # kept per trace by the page
TRAJECTORY_KEYS = ("time", *(f"{kind}_{leg}" for kind, _ in LIVE_ROWS for leg in (0, 1)))
TRAJECTORY_STRIDE = 5  # every 5th sample (20 Hz) kept for the whole experiment view
WHOLE_POINTS = 4000  # at most per trace in the whole experiment view
WHOLE_REFRESH = 1.0  # [s] between redraws of the whole experiment view
STALE_AFTER = 2.0  # [s] without samples before the stream counts as lost


# the browser's setting, for the figures
THEME_JS = """
function(_, current) {
    const theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    return theme === current ? dash_clientside.no_update : theme;
}
"""
# the page polls /live-feed itself, one request at a time (a Dash interval drops a late response when the next
# tick fires, which froze the view); it appends each chunk to the traces and shows the last LIVE_WINDOW seconds,
# or redraws a whole figure, and redraws the applied profile and the segment estimate. A zoom or pan holds the traces until a double-click.
PHASE_TRACES = [2 * row + leg for row, (kind, _) in enumerate(LIVE_ROWS) if kind == "phase" for leg in (0, 1)]
FEED_JS = (WEB / "feed.js").read_text().replace("STRIDE", str(LIVE_STRIDE)).replace("POINTS", str(FEED_POINTS)) \
    .replace("WINDOW", str(LIVE_WINDOW)).replace("PERIOD", str(FEED_MS)).replace("PHASE", str(PHASE_TRACES))


def experiment_folders(experiments: str) -> list[str]:
    """Experiment folders, newest first."""
    root = Path(experiments)
    return sorted((str(d) for d in root.iterdir() if (d / "data").is_dir()), reverse=True) if root.is_dir() else []


STEPS = ("Connect", "Walk", "Experiment")
MOVING_SPEED = 0.15  # [rad/s] the controller's still-standing threshold
BASIC_SETTINGS = ("seed", "atoms", "power_assist", "beta")  # shown, the rest under Advanced
STOP_WORDS = {"max gaits": "reached the maximum number of gaits", "seed": "power assist seed, minimum reached",
              "worse than best": "clearly worse than the best so far", "precise enough": "measured precisely enough"}


class Status(NamedTuple):
    """What the session is doing: the step reached, its tone, headline, instruction and the action offered."""

    step: int
    tone: str  # busy, action, ready, ok, error
    title: str
    detail: str = ""
    action: str | None = None  # id of the action button shown
    eyebrow: str = ""
    progress: tuple[int, int, int, str] | None = None  # valid gaits, min, max, stop reason


def trial_status(last: dict, current: dict | None, history: list[experiment.Trial], segment: dict,
                 settings: experiment.Settings) -> Status:
    """The experiment's status: starting, ramping, measuring a trial, or choosing the next profile."""
    if last["stage"] != "running":
        return Status(2, "action", "Walking stopped", "The torque is off. Start walking again to resume.", "stop",
                      "experiment paused")
    if current is None:
        return Status(2, "busy", "Starting the experiment", "Preparing the first profile. Keep walking.", "stop",
                      "experiment")
    index, rule = int(current["id"].split("-")[0]), current.get("stop_rule", {})
    eyebrow = f"trial {index} · " + ("power assist seed" if "best_mean" not in rule else "optimizing")
    if history and history[-1].index == index:
        t = history[-1]
        return Status(2, "busy", "Choosing the next profile",
                      f"Trial {index} saved: loss {t.loss:.3f} ± {t.se:.3f}. Keep walking.", "stop", eyebrow)
    if last["profile_id"] != current["id"] or last["gain"] < 1 or len(segment.get("time", [])) < 2:
        return Status(2, "busy", "Ramping up the torque", "Keep walking. Measuring starts at full torque.", "stop",
                      eyebrow)
    loss, se, gaits = experiment.gait_estimate(segment, settings)
    reason = experiment.stop_reason(loss, se, gaits, rule, settings) or ""
    detail = f"Stopping at the end of this gait: {STOP_WORDS[reason]}." if reason else "Keep walking."
    return Status(2, "ok", f"Measuring trial {index}", detail, "stop", eyebrow,
                  (gaits, settings.min_gaits, settings.max_gaits, reason))


def live_tiles(s: dict, last: dict | None, segment: dict, history: list[experiment.Trial], settings: experiment.Settings
               ) -> list[tuple]:
    """(label, value, detail) for each tile."""
    period = gait_period(s) if s else None
    gain = last["gain"] if last else None
    tiles = [
        ("gait period", f"{period:.2f} s", f"{60 / period:.0f} strides/min") if period else ("gait period", "—", "not walking"),
        ("torque", f"{100 * gain:.0f}%", "full" if gain >= 1 else "ramping" if gain > 0 else "off") if last
        else ("torque", "—", "no data"),
    ]
    _, _, gaits = experiment.gait_estimate(segment, settings) if len(segment.get("time", [])) > 1 else (0, 0, 0)
    if gaits >= 2:
        loss, se, _ = experiment.gait_estimate(segment, settings)
        tiles.append(("this trial", f"{loss:.3f}", f"± {se:.3f} · {gaits} valid gaits"))
    else:
        tiles.append(("this trial", "—", "waiting for 2 valid gaits"))
    best = min(history, key=lambda t: t.loss) if history else None
    tiles.append(("best so far", f"{best.loss:.3f}", f"trial {best.index} of {len(history)}") if best
                 else ("best so far", "—", "no trials yet"))
    return tiles


def held_label(payload: dict) -> str:
    """What a held run holds the exo on."""
    repeats = payload["repeats"]
    return dict(off="exo off", assist="power assist", best=f"best trial ({repeats})",
                trial=f"trial {repeats}")[payload["choice"]]


def held_status(last: dict, payload: dict, elapsed: float) -> Status:
    """The held run's status: holding since Start, the torque ramping to the held profile, or walking stopped."""
    eyebrow = f"held run · {held_label(payload)} · becomes trial {payload['index']}"
    clock = f"{int(elapsed // 60)}:{int(elapsed % 60):02d} held"
    if last["stage"] != "running":
        return Status(2, "action", "Walking stopped", f"{clock}. Start walking again to resume.", "stop", eyebrow)
    if last["profile_id"] != payload["id"] or last["gain"] < 1:
        return Status(2, "busy", "Ramping up the torque", f"{clock}. Keep walking.", "stop", eyebrow)
    return Status(2, "ok", "Holding the profile", f"{clock}. Stop to save it as trial {payload['index']}.", "stop",
                  eyebrow)


def launch_optimization(exp_dir: str, options: dict) -> subprocess.Popen:
    """Start a bo-loop session in the background, logging to <exp_dir>/data/bo-loop.log."""
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    flags = [arg for name, value in options.items() for arg in (f"--{name.replace('_', '-')}", str(value))]
    command = [sys.executable, "-u", "-m", "exopt.optimization_loop", "--exp-dir", exp_dir, *flags]
    with open(os.path.join(exp_dir, "data", "bo-loop.log"), "a") as log:
        # its own process group on Windows, so a ctrl-break reaches it alone
        group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else {}
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **group)


def run_dashboard(exp_dir: str, port: int, atoms: int, script: str):
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

    # drain the controller's sample stream from a background thread: a rolling buffer, and a thinned whole record
    stream, profiles = streams.driver_link()
    buffer: deque[dict] = deque(maxlen=int(BUFFER_SECONDS / 0.01))
    trajectory: deque[tuple] = deque(maxlen=int(4 * 3600 / 0.01 / TRAJECTORY_STRIDE))
    received = dict(at=-math.inf)
    # the held run now, if any, and its samples at full torque: tagged in the experiment's stream, saved as a trial
    recording: dict = dict(token=None, payload=None, started=None, samples=[])
    # every sample while an experiment is open, stamped with this computer's clock, to its stream file
    stream_log, stream_path = experiment.StreamLog(), dict(path=None)

    def pump():
        count = 0
        while True:
            sample = cast(dict, stream.recv_json())
            stream_log.write(stream_path["path"], sample | dict(wall_time=time.time()))
            # a controller restart resets its clock: drop the previous run
            if buffer and sample["time"] < buffer[-1]["time"]:
                buffer.clear()
                trajectory.clear()
            buffer.append(sample)
            if recording["payload"] and sample["profile_id"] == recording["payload"]["id"] and sample["gain"] >= 1:
                recording["samples"].append(sample)
            if count % TRAJECTORY_STRIDE == 0:
                trajectory.append(tuple(math.nan if sample[k] is None else float(sample[k]) for k in TRAJECTORY_KEYS))
            count += 1
            received["at"] = time.monotonic()

    threading.Thread(target=pump, daemon=True).start()

    # the folder and surrogate shown, and the bo-loop launched from the page, if any
    experiments = os.path.dirname(exp_dir) or "."
    # opened: an experiment is open (its folder exp_dir); held: the last held run's outcome
    run = dict(exp_dir=exp_dir, opened=False, process=None, started=None, t0=None, held=None)
    # the session shown (None: the landing page; name None: review only); hotspot: the Pi's wifi is its own network
    state: dict = dict(session=None, tunnel=None, autostart=False, hotspot=None)
    start_lock = threading.Lock()
    mock = script == MOCK

    tile = lambda label, value, sub: html.Div(className="tile", children=[
        html.Div(label, className="label"), html.Div(value, className="value"), html.Div(sub, className="sub"),
    ])
    graph = lambda id: html.Div(className="card", children=dcc.Graph(id=id, config=dict(displaylogo=False)))
    grid = lambda *ids: html.Div(className="grid", children=[graph(id) for id in ids])
    app = Dash(__name__, title="exopt", update_title=None)
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + PAGE_CSS + "</style>")
    field = lambda label, control, help="": html.Div(className="field", children=[
        html.Label(label), control, html.Div(help, className="help") if help else None])
    number = lambda id, value, width, **kw: dcc.Input(id=id, type="number", value=value, style=dict(width=width), **kw)
    defaults = experiment.DEFAULT._replace(atoms=atoms)

    def setting_field(name: str):
        label, hint = experiment.SETTINGS_HELP[name]
        default = getattr(defaults, name)
        step = 1 if isinstance(default, int) else "any"
        return field(label, number(f"exp-{name}", default, "100%", min=0, step=step), hint or "")
    hide = dict(display="none")
    saved = raspberrypi.load_settings()
    landing = html.Div(id="landing", className="landing", children=html.Div(className="card", children=[
        html.H3("Connect to the exoskeleton"),
        html.P("Simulated legs, run on this computer." if mock else
               "Over the lab wifi, the Pi's own wifi exo-pi, or a USB-C cable: whichever answers."),
        html.Div(className="controls", children=[
            html.Button("Connect", id="connect", className="primary"),
            html.Button("Review finished experiments", id="review-only"),
        ]),
        html.Details(className="advanced", style=hide if mock else {}, children=[
            html.Summary("Settings"),
            html.Div(className="form", children=[
                field("Address on the lab wifi", dcc.Input(id="pi-wifi", value=saved["wifi"], style=dict(width="100%")),
                      f"default {raspberrypi.ROUTES['wifi']}"),
                field("Password", dcc.Input(id="pi-password", type="password", value=saved["password"],
                                            style=dict(width="100%")), f"of the Pi's user {raspberrypi.USER}"),
            ]),
        ]),
    ]))
    config = html.Div(id="config", className="modal", style=hide, children=[
        html.Div(className="card", children=[
            html.H3("New experiment"),
            html.P("Each experiment gets a folder for its settings, optimization trials and everything the exo recorded. "
                   "An existing name continues that experiment. The defaults are fine for a normal session."),
            html.Div(className="form", children=[
                field("Name", dcc.Input(id="exp-name", list="exp-folders", placeholder="e.g. subject1",
                                        style=dict(width="100%")), html.Span(id="exp-folder")),
                html.Datalist(id="exp-folders"),
                *map(setting_field, BASIC_SETTINGS),
            ]),
            html.Details(className="advanced", children=[
                html.Summary("Advanced"),
                html.Div(className="form", children=[setting_field(name) for name in experiment.Settings._fields
                                                     if name not in BASIC_SETTINGS]),
            ]),
            html.Div(className="controls", style=dict(marginTop="20px"), children=[
                html.Button("Start experiment", id="start", className="primary"), html.Button("Cancel", id="cancel"),
                html.Span(id="exp-note", className="error-text"),
            ]),
        ]),
    ])
    held_config = html.Div(id="held-config", className="modal", style=hide, children=[
        html.Div(className="card", children=[
            html.H3("Run a profile"),
            html.P("The exo holds one profile until Stop (start a metabolic test together with Start here), then the "
                   "run is saved as the experiment's next trial and tagged in its recorded stream. Runs on the "
                   "experiment's power assist (exo off, when it is 0) also inform the optimization."),
            html.Div(className="form", children=[
                field("Profile", dcc.RadioItems(id="held-condition", value="off", options=[
                    dict(label=f" {label}", value=value) for value, label in optimization_loop.HELD_CHOICES.items()]),
                      html.Span(id="held-preview")),
                html.Div(id="held-trial-field", style=hide, children=field(
                    "Trial", dcc.Dropdown(id="held-trial", clearable=False, placeholder="pick a trial"))),
                field("Label", dcc.Input(id="held-label", placeholder="optional, e.g. metabolic pre",
                                         style=dict(width="100%")), "tags the run in the stream"),
            ]),
            html.Div(className="controls", style=dict(marginTop="20px"), children=[
                html.Button("Start", id="held-start", className="primary"),
                html.Button("Cancel", id="held-cancel"),
                html.Span(id="held-note", className="error-text"),
            ]),
        ]),
    ])
    # the controller's terminal, streamed live beside the status
    console = html.Div(className="console", children=[
        html.Div(id="console-summary", className="console-head"),
        html.Div(className="log-scroll", children=html.Pre(id="controller-log", className="log")),
        html.Div(className="controls", children=[
            html.Button("Stop controller", id="controller-stop"),
            html.Span("Sends ctrl-c: the drives are released.", className="hint"),
        ]),
    ])
    status_section = html.Div(id="status", className="status tone-busy", children=[
        html.Ol(id="stepper", className="stepper"),
        html.Div(className="status-row", children=[html.Div(className="hero", children=[
            html.Div(className="hero-text", children=[
                html.Div(id="hero-eyebrow", className="eyebrow"), html.Div(id="hero-title", className="title"),
                html.Div(id="hero-detail", className="detail"), html.Div(id="hero-progress"),
            ]),
            html.Div(className="actions", children=[
                html.Button("Start experiment", id="configure", className="primary", style=hide),
                html.Div(className="stack", children=[
                    html.Button("Start optimization", id="optimize", className="primary", style=hide),
                    html.Button("Run a profile", id="held", className="primary", style=hide),
                    html.Button("Close experiment", id="close-experiment", style=hide),
                ]),
                html.Button("Restart controller", id="controller-start", className="primary", style=hide),
                html.Button("Reconnect", id="reconnect", className="primary", style=hide),
                html.Button("Stop optimization", id="stop", className="danger", style=hide),
            ]),
        ]), console]),
    ])
    live_section = html.Div(id="live-section", children=[
        html.Div(id="tiles", className="tiles"),
        html.Div(className="grid", children=[html.Div(className="card", children=html.Div(id=id))
                                             for id in ("now-profile", "segment")]),
        html.Div(className="section-head", children=[
            html.H2("Leg signals"),
            dcc.RadioItems(id="live-span", value="window", inline=True, className="segmented", options=[
                dict(label=f"last {LIVE_WINDOW:g} s", value="window"), dict(label="whole experiment", value="whole")]),
            html.Span(id="live-hint", className="hint"),
        ]),
        # plain divs, plotted by FEED_JS alone: a dcc.Graph replots its own (empty) figure on every relayout
        html.Div(className="grid", children=html.Div(className="card", children=html.Div(id="live"))),
    ])
    review_section = html.Div(id="review", style=hide, children=[
        html.Div(className="section-head", children=[
            html.H2("Results"),
            dcc.Dropdown(id="review-dir", clearable=False, style=dict(width="300px")),
            html.Span(id="review-headline", className="hint"),
        ]),
        grid(*CLICKABLE),
        html.Div(className="section-head", children=[
            html.H2("Trial details"), html.Span(id="summary", className="hint"),
        ]),
        html.Div(className="hint", style=dict(margin="-4px 0 8px"),
                 children="Click a point in “loss per trial” or a line in “profiles tried” to inspect another trial."),
        grid(*(f"trial-{name}" for name in TRIAL)),
        html.Div(className="section-head", children=[
            html.H2("Whole experiment"),
            html.Span("leg signals, shaded by trial and by the transitions between them", className="hint"),
        ]),
        grid("experiment"),
    ])
    app.layout = html.Div(className="page", children=[
        html.Header(className="topbar", children=[
            html.Span("exopt", className="brand"),
            html.Span(id="session-pill", className="pill", style=hide),
            html.Span(id="experiment-pill", className="pill", style=hide),
            html.Span(className="spacer"),
            # over wifi the switch drops the connection: reconnect once this computer joined the other network
            dcc.ConfirmDialogProvider(id="wifi-switch", children=html.Button(id="wifi-switch-label")),
            html.Button("Disconnect", id="disconnect", style=hide),
        ]),
        landing,
        html.Div(id="session", style=hide, children=[status_section, live_section, review_section]),
        config, held_config,
        dcc.Store(id="selected"), dcc.Store(id="latest-trial"),
        dcc.Store(id="theme"),
        dcc.Store(id="live-loop"),
        dcc.Interval(id="fast", interval=REFRESH_MS),
        dcc.Interval(id="slow", interval=2500),
    ])

    app.clientside_callback(FEED_JS, Output("live-loop", "data"), Input("theme", "data"), Input("live-span", "value"))

    def live_payload(s: dict, p: Palette, stride: int, window: bool) -> dict:
        fig = live_figure(s, p, stride)
        fig = json.loads(fig.to_json()) if isinstance(fig, go.Figure) else fig
        fig["layout"].pop("uirevision", None)  # the page holds a zoomed view itself
        latest = float(s["time"][-1]) if s else None
        if window and latest is not None:
            fig["layout"]["xaxis"]["range"] = [latest - LIVE_WINDOW, latest]
        return dict(figure=fig, latest=latest)

    def trace_update(data: list[dict], p: Palette, span: str, cursor: float | None, paused: bool) -> dict | None:
        """New samples for the page's traces; the whole figure when it has none, or the view, theme or controller changed."""
        full = cursor is None or bool(data and data[-1]["time"] < cursor)
        if paused and not full:
            return None

        # the whole experiment (else the whole controller run), thinned, redrawn every WHOLE_REFRESH seconds
        if span == "whole":
            if not full and (not data or data[-1]["time"] - cursor < WHOLE_REFRESH):
                return None
            rows, t0 = list(trajectory), run["t0"]
            if t0 is not None and rows and rows[0][0] <= t0 <= rows[-1][0]:
                rows = [r for r in rows if r[0] >= t0]
            rows = rows[::max(1, math.ceil(len(rows) / WHOLE_POINTS))]
            return live_payload(dict(zip(TRAJECTORY_KEYS, np.array(rows).T)) if rows else {}, p, 1, window=False)

        if full:
            kept = [x for x in data if x["time"] > data[-1]["time"] - LIVE_WINDOW] if data else []
            return live_payload(columns(kept), p, LIVE_STRIDE, window=True)
        fresh = columns([x for x in data if x["time"] > cursor])
        if not fresh:
            return None
        traces = [cycles(fresh[f"{kind}_{leg}"]) if kind == "phase" else fresh[f"{kind}_{leg}"]
                  for kind, _ in LIVE_ROWS for leg in (0, 1)]
        return dict(t=fresh["time"].tolist(), y=[y.tolist() for y in traces], latest=float(fresh["time"][-1]))

    as_dict = lambda figure: json.loads(go.Figure(figure).to_json()) if isinstance(figure, go.Figure) else figure
    profile_bases: dict[tuple, dict] = {}  # the applied profile's figure without samples, per profile and theme

    def panel_updates(data: list[dict], p: Palette, theme: str, shown: dict) -> dict:
        """The figures of the applied profile against the live torque, and of the segment estimate, that changed."""
        titles = {"now-profile": "torque profile now applied", "segment": "this trial's loss estimate"}
        if not (state["session"] or {}).get("name"):
            return {id: dict(key="none", figure=as_dict(placeholder(p, "no controller", title)))
                    for id, title in titles.items() if shown.get(id) != "none"}
        current = experiment.load_current(run["exp_dir"])
        current_id = current and current["id"]
        latest = data[-1]["time"] if data else None
        figures = {}

        # the profile with the last TORQUE_WINDOW seconds of commanded torque over the gait phase
        key = f"{current_id}:{latest}" if current or data else "idle"
        if shown.get("now-profile") != key:
            recent = columns([x for x in data if x["time"] > latest - TORQUE_WINDOW]) if data else {}
            if not recent and current is None:
                figure = as_dict(placeholder(p, "no profile running", titles["now-profile"]))
            else:
                if (current_id, theme) not in profile_bases:
                    empty = {f"{kind}_{leg}": np.zeros(0) for kind in ("phase", "torque") for leg in (0, 1)}
                    profile = experiment.decode_profile(current) if current else None
                    profile_bases.clear()
                    profile_bases[current_id, theme] = as_dict(
                        profile_figure(titles["now-profile"], profile, empty, "torque",
                                       power_assist=(current or {}).get("power_assist", 0.0), p=p))
                base = profile_bases[current_id, theme]
                points = []
                for leg in (0, 1):
                    phase = cycles(recent[f"phase_{leg}"]) if recent else np.zeros(0)
                    x, y = break_wraps(phase, recent[f"torque_{leg}"] if recent else np.zeros(0), phase)
                    points.append(base["data"][leg] | dict(x=x.tolist(), y=y.tolist(), mode="lines", opacity=0.6,
                                                           line=dict(color=p.legs[leg], width=1.5)))
                figure = dict(data=points + base["data"][2:], layout=base["layout"])
            figures["now-profile"] = dict(key=key, figure=figure)

        # samples of the profile being collected, once at full gain, after the warmup
        settings = experiment.load_settings(run["exp_dir"])
        segment = [x for x in data if current and x["profile_id"] == current_id and x["gain"] >= 1][settings.warmup:]
        key = f"{current_id}:{len(segment)}"
        if shown.get("segment") != key:
            figure = estimate_figure(titles["segment"], columns(segment), settings, current and current.get("stop_rule"),
                                     p=p)
            figures["segment"] = dict(key=key, figure=as_dict(figure))
        return figures

    @app.server.route("/live-feed")
    def live_feed():
        from flask import request
        from plotly.io.json import to_json_plotly
        # plotly's encoder writes NaN (no phase yet, dropped samples) as null, which JSON can carry
        jsonify = lambda **reply: app.server.response_class(to_json_plotly(reply), mimetype="application/json")
        arg, data = request.args.get, list(buffer)
        theme = arg("theme", "light")
        p = THEMES.get(theme, LIGHT)
        panels = panel_updates(data, p, theme, {id: arg(id, "") for id in ("now-profile", "segment")})
        if not (state["session"] or {}).get("name"):
            return jsonify(panels=panels, idle=True)
        cursor = float(arg("cursor")) if arg("cursor") else None
        return jsonify(traces=trace_update(data, p, arg("span", "window"), cursor, arg("paused") == "1"),
                       panels=panels, idle=False)

    app.clientside_callback(
        THEME_JS, Output("theme", "data"),
        Input("fast", "n_intervals"), State("theme", "data"),
    )

    def experiment_running() -> bool:
        return run["process"] is not None and run["process"].poll() is None

    def recording_active() -> bool:
        return recording["token"] is not None

    def busy() -> bool:
        return experiment_running() or recording_active()

    def hold(token: object, folder: str):
        """Hold the profile (republished each second, the controller ignores repeats), then release, save the run as
        a trial (from the warm-up to the last full gait) and tag the stop."""
        path = stream_path["path"]
        while recording["token"] is token:
            profiles.send_json(recording["payload"])
            time.sleep(1.0)
        streams.release(profiles)
        payload, settings = recording["payload"], experiment.load_settings(folder)
        segment = experiment.held_segment(recording["samples"], settings)
        gaits = len(experiment.gait_slices(experiment.columns(segment), settings)) if segment else 0
        saved = None
        if gaits >= 2:
            experiment.save(folder, payload["index"], experiment.decode_profile(payload), segment, settings,
                        payload["power_assist"])
            saved = payload["index"]
        run.update(held=f"Saved as trial {saved} ({gaits} valid gaits)." if saved else
                   f"Not saved: {gaits} valid gaits, a trial needs 2.")
        stream_log.event(path, dict(held="stop", trial=saved, gaits=gaits,
                                    stopped_at=datetime.now().astimezone().isoformat()))

    def start_recording(folder: str, choice: str, trial: int | None, label: str):
        profile, power_assist, repeats = held_mode(choice, trial)[1]
        settings = experiment.load_settings(folder)
        index = max((t.index for t in load_trials(folder)), default=0) + 1
        payload = experiment.encode_profile(profile) | dict(
            id=f"{index}-{uuid.uuid4().hex[:6]}", ramp_cycles=settings.ramp_cycles, power_assist=power_assist,
            stop_rule={}, index=index, choice=choice, repeats=repeats, label=label.strip())
        experiment.save_current(folder, payload)
        token = object()
        recording.update(token=token, payload=payload, started=time.monotonic(), samples=[])
        stream_log.event(stream_path["path"], dict(
            held="start", choice=choice, label=label.strip(), trial=index, repeats=repeats,
            profile=payload, started_at=datetime.now().astimezone().isoformat(),
        ))
        run.update(exp_dir=folder, t0=buffer[-1]["time"] if buffer else None, held="")
        threading.Thread(target=hold, args=(token, folder), daemon=True).start()

    def open_session(wifi: str | None = None, password: str | None = None):
        """Open the session: the tunnel to the Pi over the route that answers (or this computer's mock), then its
        controller. An address and password given are saved as this computer's."""
        if wifi is not None:
            raspberrypi.save_settings(wifi.strip() or raspberrypi.ROUTES["wifi"], password or "")
        buffer.clear()
        trajectory.clear()
        settings = raspberrypi.load_settings()
        opened = raspberrypi.Local() if mock else raspberrypi.Tunnel(raspberrypi.ROUTES | dict(wifi=settings["wifi"]), settings["password"],
                                                   (streams.SAMPLE_PORT, streams.PROFILE_PORT))
        state.update(session=dict(name="mock" if mock else "pi"), tunnel=opened, autostart=True, hotspot=None)

    def close_session():
        if state["tunnel"] is not None:
            state["tunnel"].close()  # ctrl-c to the controller first
        state.update(session=None, tunnel=None, autostart=False, hotspot=None)

    def start_controller():
        """Run the controller script, on the Pi from this computer's copy of the code."""
        tunnel = state["tunnel"]
        if not mock:
            try:
                state["hotspot"] = tunnel.hotspot_on()
            except Exception:  # no switch offered
                state["hotspot"] = None
        return tunnel.start(script)

    @app.callback(
        Output("landing", "style"), Output("session", "style"), Output("disconnect", "style"),
        Output("session-pill", "style"), Output("experiment-pill", "style"),
        Output("status", "style"), Output("live-section", "style"), Output("disconnect", "children"),
        Input("connect", "n_clicks"), Input("review-only", "n_clicks"), Input("disconnect", "n_clicks"),
        Input("fast", "n_intervals"), State("pi-wifi", "value"), State("pi-password", "value"),
    )
    def session(_connect, _review, _disconnect, _, wifi, password):
        """The landing page, or the session (a reloaded page returns to the open one)."""
        if ctx.triggered_id == "connect" and state["session"] is None:
            open_session(wifi or "", password)
        elif ctx.triggered_id == "review-only" and state["session"] is None:
            state.update(session=dict(name=None))
        elif ctx.triggered_id == "disconnect" and not busy():
            close_session()
        if state["session"] is None:
            return dict(), hide, hide, hide, hide, no_update, no_update, no_update
        if state["session"]["name"] is None:  # review only: no controller, no live view
            return hide, dict(), dict(), hide, hide, hide, hide, "Back"
        return hide, dict(), dict(), dict(), dict(), dict(), dict(), "Disconnect"

    @app.callback(
        Output("wifi-switch-label", "children"), Output("wifi-switch-label", "style"),
        Output("wifi-switch-label", "disabled"), Output("wifi-switch", "message"),
        Input("wifi-switch", "submit_n_clicks"), Input("slow", "n_intervals"),
    )
    def wifi_switch(_submit, _):
        """Move the Pi's wifi between the lab's and its own network, once the session knows which it is on."""
        tunnel, hotspot = state["tunnel"], state["hotspot"]
        if mock or tunnel is None or hotspot is None or not (tunnel.connected and tunnel.alive()):
            return no_update, hide, no_update, no_update
        if ctx.triggered_id == "wifi-switch" and not busy():
            try:
                tunnel.switch_wifi(not hotspot)
                hotspot = state["hotspot"] = not hotspot
            except Exception as error:
                print(f"wifi switch failed: {error}")
        dropped = "" if tunnel.route == "usb" else (" This connection drops: join the other network on this computer, "
                                                    "then Reconnect.")
        if hotspot:
            return "Pi wifi: exo-pi", dict(), busy(), ("Turn the Pi's own wifi exo-pi off, so it joins the lab wifi "
                                                       "(exo-pi comes back if it can't within 30 s)?" + dropped)
        return "Pi wifi: lab", dict(), busy(), "Switch the Pi to its own wifi network exo-pi?" + dropped

    def named_folder(name: str) -> str:
        """An existing experiment folder of that name, else today's new one."""
        name = name.strip()
        existing = os.path.join(experiments, name)
        if name and os.path.isdir(existing):
            return existing
        return os.path.join(experiments, f"{time.strftime('%Y-%m-%d')}-{name.replace(' ', '-') or '<name>'}")

    def folder_options() -> list:
        return [html.Option(value=os.path.basename(f)) for f in experiment_folders(experiments)]

    @app.callback(
        Output("config", "style"), Output("exp-note", "children"), Output("exp-folders", "children"),
        Input("configure", "n_clicks"), Input("start", "n_clicks"), Input("cancel", "n_clicks"),
        State("exp-name", "value"), prevent_initial_call=True,
    )
    def config_window(_configure, _start, _cancel, name):
        if ctx.triggered_id == "configure":
            return dict(display="flex"), "", folder_options()
        if ctx.triggered_id == "start" and not (name and name.strip()):
            return no_update, "give the experiment a name", no_update
        return hide, "", no_update

    @app.callback(Output("exp-folder", "children"), *(Output(f"exp-{name}", "value") for name in experiment.Settings._fields),
                  Input("exp-name", "value"))
    def experiment_folder(name):
        folder = named_folder(name or "")
        if os.path.isdir(folder):
            return f"continues {folder} ({len(load_trials(folder))} trials)", *experiment.load_settings(folder)
        return f"saved to {folder}", *(no_update for _ in experiment.Settings._fields)

    held_profiles: dict[tuple, tuple] = {}  # per folder, choice, trial and trial count: the best takes a fit

    def held_mode(choice: str, trial: int | None) -> tuple[str, tuple | None, str]:
        """The open experiment's folder and the profile a held run holds (optimization_loop.held_profile); else why not."""
        folder = run["exp_dir"]
        history = load_trials(folder)
        if choice == "best" and not optimization_loop.fitted(history, experiment.load_settings(folder)):
            return folder, None, f"{os.path.basename(folder)} has no optimization trials to take the best from"
        if choice == "trial" and trial not in {t.index for t in history}:
            return folder, None, "pick a trial"
        key = (folder, choice, trial if choice == "trial" else None, len(history))
        if key not in held_profiles:
            held_profiles[key] = optimization_loop.held_profile(folder, choice, trial)
        return folder, held_profiles[key], ""

    @app.callback(
        Output("held-config", "style"), Output("held-note", "children"),
        Input("held", "n_clicks"), Input("held-start", "n_clicks"), Input("held-cancel", "n_clicks"),
        State("held-condition", "value"), State("held-trial", "value"), prevent_initial_call=True,
    )
    def held_window(_open, _start, _cancel, choice, trial):
        if ctx.triggered_id == "held":
            return dict(display="flex"), ""
        if ctx.triggered_id == "held-start" and (problem := held_mode(choice, trial)[2]):
            return no_update, problem
        return hide, ""

    @app.callback(
        Output("held-preview", "children"), Output("held-trial-field", "style"), Output("held-trial", "options"),
        Input("held", "n_clicks"), Input("held-condition", "value"), Input("held-trial", "value"),
        prevent_initial_call=True,
    )
    def held_preview(_open, choice, trial):
        """What Start will hold, worked out ahead (the best trial's fit takes seconds), so Start is instant."""
        folder, profile, problem = held_mode(choice, trial)
        history = load_trials(folder)
        options = [dict(label=f"trial {t.index} · loss {t.loss:.3f}", value=t.index) for t in history]
        trial_style = dict() if choice == "trial" else hide
        if problem:
            return problem, trial_style, options
        _, power_assist, repeats = profile
        index = max((t.index for t in history), default=0) + 1
        repeated = f"repeats trial {repeats}, " if repeats is not None else ""
        fits = "" if power_assist == experiment.load_settings(folder).power_assist else ", kept out of the optimization"
        return f"{repeated}saved as trial {index}{fits}", trial_style, options

    def session_status(last: dict | None, current: dict | None, history: list[experiment.Trial], segment: dict,
                       window: dict, settings: experiment.Settings) -> Status:
        """Where the session is, from the tunnel, the controller, its stream and the experiment."""
        tunnel = state["tunnel"]
        name = "the Pi" if state["session"]["name"] == "pi" else "the mock"
        command = tunnel.command if tunnel.alive() else None
        if not tunnel.alive():
            return Status(0, "error", f"Can't reach {name}", tunnel.error or "The ssh connection dropped.", "reconnect")
        if not tunnel.connected:
            return Status(0, "busy", f"Connecting to {name}…",
                          tunnel.target if tunnel.route else "Trying the lab wifi, its own wifi exo-pi and the USB-C cable.")
        if last is None:
            if command is not None and command.running():
                if received["at"] > -math.inf:
                    return Status(0, "error", "No data from the controller",
                                  "It stopped streaming. Check its output below.", "controller-start")
                return Status(0, "busy", "Starting the controller…", "This takes a few seconds.")
            output = (command.output.strip().split("\n") or [""])[-1] if command is not None else ""
            return Status(0, "error", "The controller stopped", output or "Restart it to continue.",
                          "controller-start")
        if recording_active():
            return held_status(last, recording["payload"], time.monotonic() - recording["started"])
        if experiment_running():
            return trial_status(last, current, history, segment, settings)
        recent = window["time"] > window["time"][-1] - 2.0 if window else []
        moving = window and any(np.nanmax(np.abs(window[f"velocity_{leg}"][recent])) > MOVING_SPEED for leg in (0, 1))
        # the experiment can be set up while the gait calibrates
        configure = None if run["opened"] else "configure"
        if last["stage"] == "calibrating" or (last["stage"] == "standing" and moving):
            return Status(1, "busy", "Keep walking…", "Calibrating the gait: it takes a few strides. The torque "
                          "stays off until then.", configure)
        if last["stage"] == "standing":
            return Status(1, "action", "Start walking", "The gait is calibrated after a few strides; the torque "
                          "stays off until then.", configure)
        if not run["opened"]:
            return Status(2, "ready", "Ready", "Start an experiment: a folder for the subject's settings, "
                          "optimization trials and held runs.", "configure")
        ended = f" The optimization stopped after {len(history)} trials." if run["process"] is not None else ""
        if run["held"] is not None:
            ended = " " + (run["held"] or "Saving the held run…")
        return Status(2, "ready", os.path.basename(run["exp_dir"]), "Start the optimization, where profiles are tried "
                      "one after the other while the subject walks, or run a chosen profile, held until Stop."
                      + ended, "optimize", f"experiment · {len(history)} trials")

    def progress_bar(gaits: int, low: int, high: int, reason: str):
        fill = min(gaits / high, 1.0) * 100
        return html.Div(className="progress", children=[
            html.Div(className="bar", children=[
                html.Div(className="fill", style=dict(width=f"{fill:.1f}%")),
                html.Div(className="min", style=dict(left=f"{100 * low / high:.1f}%"), title="minimum"),
            ]),
            html.Div(className="caption", children=[
                html.Span(f"{gaits} valid gaits"), html.Span(f"stops between {low} and {high}"),
            ]),
        ])

    actions = ("configure", "optimize", "held", "close-experiment", "controller-start", "reconnect", "stop")

    @app.callback(
        Output("status", "className"), Output("stepper", "children"), Output("hero-eyebrow", "children"),
        Output("hero-title", "children"), Output("hero-detail", "children"), Output("hero-progress", "children"),
        *(Output(id, "style") for id in actions),
        Output("controller-log", "children"), Output("console-summary", "children"),
        Output("controller-stop", "disabled"), Output("tiles", "children"),
        Output("session-pill", "children"), Output("experiment-pill", "children"), Output("disconnect", "disabled"),
        Output("stop", "children"),
        Input("controller-start", "n_clicks"), Input("controller-stop", "n_clicks"),
        Input("start", "n_clicks"), Input("optimize", "n_clicks"), Input("close-experiment", "n_clicks"),
        Input("stop", "n_clicks"), Input("reconnect", "n_clicks"),
        Input("held-start", "n_clicks"), Input("fast", "n_intervals"),
        State("held-condition", "value"), State("held-trial", "value"), State("held-label", "value"),
        State("exp-name", "value"), *(State(f"exp-{name}", "value") for name in experiment.Settings._fields),
    )
    def status(*args):
        """Act on a click, then render the session from server state (so a dropped response heals on the next tick)."""
        fields = len(experiment.Settings._fields)
        name, values = args[-1 - fields], args[-fields:]
        choice, held_trial, label = args[-4 - fields:-1 - fields]
        if state["session"] is None or state["session"]["name"] is None:
            return (no_update,) * 21
        tunnel, trigger = state["tunnel"], ctx.triggered_id

        # clicks: reconnect, the controller, the experiment
        if trigger == "reconnect" and not busy():
            close_session()
            open_session()
            tunnel = state["tunnel"]
        connected = tunnel.connected and tunnel.alive()
        # ticks run concurrently: without the lock two of them autostart two controllers (racing clones)
        with start_lock:
            command = tunnel.command
            running = command is not None and command.running()
            if (trigger == "controller-start" or (state["autostart"] and command is None)) and connected:
                if running:
                    command.interrupt()
                command = start_controller()
                running, state["autostart"] = True, False
        if trigger == "controller-stop" and running:
            command.interrupt()
        elif trigger == "start" and not busy() and name and name.strip():
            # a cleared field keeps its default; bo-loop clamps the gait counts
            folder = named_folder(name)
            experiment.save_settings(folder, experiment.Settings(*(default if v is None else type(default)(v)
                                                           for default, v in zip(defaults, values))))
            run.update(exp_dir=folder, opened=True, process=None, held=None)
            stream_path["path"] = Path(folder) / "data" / experiment.STREAM
        elif trigger == "close-experiment" and not busy():
            run.update(opened=False, process=None, held=None)
            stream_path["path"] = None
        elif trigger == "optimize" and run["opened"] and not busy():
            options = experiment.load_settings(run["exp_dir"])._asdict()
            run.update(process=launch_optimization(run["exp_dir"], options), started=time.monotonic(),
                       t0=buffer[-1]["time"] if buffer else None, held=None)
        elif trigger == "held-start" and run["opened"] and not busy():
            folder, _, problem = held_mode(choice, held_trial)
            if not problem:
                start_recording(folder, choice, held_trial, label or "")
        elif trigger == "stop" and recording_active():
            # the holding thread releases the exo, saves the trial and tags the stop
            recording["token"] = None
        elif trigger == "stop" and experiment_running():
            # an interrupt, so bo-loop releases the profile and the torque ramps down
            run["process"].send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)

        # what is going on
        data = list(buffer)
        last = data[-1] if data and time.monotonic() - received["at"] < STALE_AFTER else None
        window = columns([x for x in data if x["time"] > data[-1]["time"] - LIVE_WINDOW]) if data else {}
        current = experiment.load_current(run["exp_dir"]) if busy() else None
        history, settings = load_trials(run["exp_dir"]), experiment.load_settings(run["exp_dir"])
        segment = columns([x for x in data if current and x["profile_id"] == current["id"] and x["gain"] >= 1]
                          [settings.warmup:])
        now = session_status(last, current, history, segment, window, settings)

        # the steps, the headline and its one action
        steps = []
        for k, step_name in enumerate(STEPS):
            kind = "done" if k < now.step else ("active" if k == now.step else "todo")
            steps.append(html.Li(className=f"step {kind}", children=[
                html.Span("✓" if kind == "done" else str(k + 1), className="num"), html.Span(step_name)]))
        offered = {now.action, *(("held", "close-experiment") if now.action == "optimize" else ()),
                   *(("stop",) if busy() else ())}
        shown = lambda id: dict() if id in offered else hide

        # the controller's shell, folded away
        log = "\n".join(command.output.rstrip("\n").split("\n")[-200:]) if command is not None else ""
        if command is not None and not running:
            exit_status = command.exit_status()
            log += "\n· controller exited" + (f" ({exit_status})" if exit_status not in (None, -1) else "")
        summary = f"Controller terminal · live · {'running' if running else 'stopped'} · {script}"

        dot = {"error": "error", "busy": "busy"}.get(now.tone, "ok") if now.step or now.tone != "busy" else "busy"
        pill = [html.Span(className=f"dot {dot}"), html.B(state["session"]["name"]), tunnel.target]
        experiment_tag = [html.Span("experiment"), html.B(os.path.basename(run["exp_dir"]))] if run["opened"] else []
        return (
            f"status tone-{now.tone}", steps, now.eyebrow or f"step {now.step + 1} of {len(STEPS)}", now.title,
            now.detail, progress_bar(*now.progress) if now.progress else None,
            *(shown(id) for id in actions), log, summary, not running,
            [tile(*t) for t in live_tiles(window, last, segment, history, settings)], pill, experiment_tag, busy(),
            "Stop and save the trial" if recording_active() else "Stop optimization",
        )

    @app.callback(
        Output("review-dir", "options"), Output("review-dir", "value"), Output("review-headline", "children"),
        Output("review", "style"), Input("slow", "n_intervals"), Input("review-dir", "value"),
    )
    def review_folder(_, folder):
        folders = experiment_folders(experiments)
        # a run launched from here takes over the review, as does the first folder on opening
        if run["exp_dir"] != run.get("reviewed") and os.path.isdir(run["exp_dir"]):
            folder = run["reviewed"] = run["exp_dir"]
        elif folder is None:
            folder = folders[0] if folders else None
        options = [dict(label=os.path.basename(f), value=f) for f in sorted({*folders, *([folder] if folder else [])},
                                                                           reverse=True)]
        history = load_trials(folder) if folder else []
        headline = review_headline(history) if history else "no trials (or an older format)" if folder else ""
        return options, no_update if ctx.triggered_id == "review-dir" else folder, headline, \
            dict() if history or state["session"] == dict(name=None) else hide

    @app.callback(
        Output("selected", "data"), Output("latest-trial", "data"),
        Input("slow", "n_intervals"), Input("review-dir", "value"), *(Input(id, "clickData") for id in CLICKABLE),
        State("selected", "data"), State("latest-trial", "data"),
    )
    def select(_, folder, *clicks_and_state):
        *clicks, selected, shown_latest = clicks_and_state
        index = [t.index for t in load_trials(folder)] if folder else []
        latest = index[-1] if index else None
        trigger, choice = ctx.triggered_id, selected

        # a click on any trial marker or line
        if trigger in CLICKABLE:
            data = ((clicks[CLICKABLE.index(trigger)] or {}).get("points") or [{}])[0].get("customdata")
            choice = int(data[0] if isinstance(data, list) else data) if data is not None else selected
        # on the latest trial, the selection moves on with each new one
        elif trigger == "review-dir" or selected not in index or selected == shown_latest:
            choice = latest

        return no_update if choice == selected else choice, no_update if latest == shown_latest else latest

    @app.callback(
        *(Output(id, "figure") for id in CLICKABLE),
        Input("slow", "n_intervals"), Input("selected", "data"), Input("theme", "data"), State("review-dir", "value"),
    )
    def refresh_optimization(_, selected, theme, folder):
        history, p = load_trials(folder) if folder else [], THEMES.get(theme, LIGHT)
        figures = (loss_figure, profiles_figure)
        return tuple(figure(history, selected, p=p) for figure in figures)

    @app.callback(
        *(Output(f"trial-{name}", "figure") for name in TRIAL), Output("summary", "children"),
        Input("selected", "data"), Input("slow", "n_intervals"), Input("theme", "data"), State("review-dir", "value"),
    )
    def refresh_trial(selected, _, theme, folder):
        trial = next((t for t in (load_trials(folder) if folder else []) if t.index == selected), None)
        p = THEMES.get(theme, LIGHT)
        if trial is None:
            return (placeholder(p, "click a trial above"),) * 4 + ("",)
        return (
            profile_figure("profile and its atoms", trial.profile, trial.samples, "measured_torque", atoms=True,
                           power_assist=trial.power_assist, p=p),
            power_figure(trial, p=p),
            estimate_figure("convergence of the estimate", trial.samples, trial.settings, p=p),
            portrait_figure(trial, p=p),
            trial_summary(trial),
        )

    shown_experiment: dict = dict(key=None)  # what the whole experiment figure shows: redrawn with each new trial

    @app.callback(
        Output("experiment", "figure"),
        Input("review-dir", "value"), Input("slow", "n_intervals"), Input("theme", "data"),
    )
    def refresh_experiment(folder, _, theme):
        history, p = load_trials(folder) if folder else [], THEMES.get(theme, LIGHT)
        key = (folder, theme, tuple(t.index for t in history))
        if key == shown_experiment["key"] and ctx.triggered_id == "slow":
            return no_update
        shown_experiment["key"] = key
        return experiment_figure(folder, history, p) if folder else placeholder(p, "no experiment", height=LIVE_HEIGHT)

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock", help="experiment folder")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--atoms", type=int, default=experiment.DEFAULT.atoms, help="new experiment default (as bo-loop)")
    parser.add_argument("--mock", action="store_true", help="simulated legs on this computer instead of the Pi")
    parser.add_argument("--diagnose", action="store_true", help="run the drive check instead of the controller")
    args = parser.parse_args()

    run_dashboard(args.exp_dir, args.port, args.atoms,
                  DIAGNOSE if args.diagnose else MOCK if args.mock else CONTROLLER)


if __name__ == "__main__":
    main()
