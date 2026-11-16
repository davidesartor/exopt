"""start: connect to the exo, run an experiment, review any experiment."""

from typing import NamedTuple, cast
from collections import deque
from pathlib import Path
import argparse
import json
import math
import os
import posixpath
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser

import numpy as np
import paramiko
import plotly.graph_objects as go

from exopt import link, trials
from exopt.figures import (CLICKABLE, LIGHT, LIVE_ROWS, LIVE_STRIDE, LIVE_WINDOW, PAGE_CSS, THEMES, TRIAL, WEB, Palette,
                           break_wraps, cycles, estimate_figure, gait_period, live_figure,
                           load_trials, loss_figure, placeholder, portrait_figure, power_figure, profile_figure,
                           profiles_figure, review_headline, trial_summary)
from exopt.tunnel import (SSH_CONFIG, Tunnel, delete_controller, known_hosts, save_controller, save_host,
                          saved_controllers)

columns = trials.columns
BUFFER_SECONDS = 60.0  # [s] of samples kept at full rate
TORQUE_WINDOW = 4.0  # [s] of live torque overlaid on the applied profile
NEW = "::new"  # the controller picker's configure new entry
REPO = Path(__file__).resolve().parents[2]
CONTROLLER, MOCK_CONTROLLER, DIAGNOSE = "src/exopt/controller.py", "src/exopt/mock_controller.py", "src/exopt/diagnose.py"
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
BASIC_SETTINGS = ("seed", "atoms", "power_assist", "beta", "n")  # shown, the rest under Advanced
STOP_WORDS = {"max gaits": "reached the maximum number of gaits", "initial design": "initial-design profile, minimum reached",
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


def trial_status(last: dict, current: dict | None, history: list[trials.Trial], segment: dict,
                 settings: trials.Settings) -> Status:
    """The experiment's status: starting, ramping, measuring a trial, or choosing the next profile."""
    if last["stage"] != "running":
        return Status(2, "action", "Walking stopped", "The torque is off. Start walking again to resume.", "stop",
                      "experiment paused")
    if current is None:
        return Status(2, "busy", "Starting the experiment", "Preparing the first profile. Keep walking.", "stop",
                      "experiment")
    index, rule = int(current["id"].split("-")[0]), current.get("stop_rule", {})
    eyebrow = f"trial {index} · " + ("initial design" if "best_mean" not in rule else "optimizing")
    if history and history[-1].index == index:
        t = history[-1]
        return Status(2, "busy", "Choosing the next profile",
                      f"Trial {index} saved: loss {t.loss:.3f} ± {t.se:.3f}. Keep walking.", "stop", eyebrow)
    if last["profile_id"] != current["id"] or last["gain"] < 1 or len(segment.get("time", [])) < 2:
        return Status(2, "busy", "Ramping up the torque", "Keep walking. Measuring starts at full torque.", "stop",
                      eyebrow)
    loss, se, gaits = trials.gait_estimate(segment, settings)
    reason = trials.stop_reason(loss, se, gaits, rule, settings) or ""
    detail = f"Stopping at the end of this gait: {STOP_WORDS[reason]}." if reason else "Keep walking."
    return Status(2, "ok", f"Measuring trial {index}", detail, "stop", eyebrow,
                  (gaits, settings.min_gaits, settings.max_gaits, reason))


def live_tiles(s: dict, last: dict | None, segment: dict, history: list[trials.Trial], settings: trials.Settings
               ) -> list[tuple]:
    """(label, value, detail) for each tile."""
    period = gait_period(s) if s else None
    gain = last["gain"] if last else None
    tiles = [
        ("gait period", f"{period:.2f} s", f"{60 / period:.0f} strides/min") if period else ("gait period", "—", "not walking"),
        ("torque", f"{100 * gain:.0f}%", "full" if gain >= 1 else "ramping" if gain > 0 else "off") if last
        else ("torque", "—", "no data"),
    ]
    _, _, gaits = trials.gait_estimate(segment, settings) if len(segment.get("time", [])) > 1 else (0, 0, 0)
    if gaits >= 2:
        loss, se, _ = trials.gait_estimate(segment, settings)
        tiles.append(("this trial", f"{loss:.3f}", f"± {se:.3f} · {gaits} valid gaits"))
    else:
        tiles.append(("this trial", "—", "waiting for 2 valid gaits"))
    best = min(history, key=lambda t: t.loss) if history else None
    tiles.append(("best so far", f"{best.loss:.3f}", f"trial {best.index} of {len(history)}") if best
                 else ("best so far", "—", "no trials yet"))
    return tiles


def launch_bo_loop(exp_dir: str, host: str, options: dict) -> subprocess.Popen:
    """Start a bo-loop session in the background, logging to <exp_dir>/data/bo-loop.log."""
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    flags = [arg for name, value in options.items() for arg in (f"--{name.replace('_', '-')}", str(value))]
    command = [sys.executable, "-u", "-m", "exopt.bo_loop", "--exp-dir", exp_dir, "--host", host, *flags]
    with open(os.path.join(exp_dir, "data", "bo-loop.log"), "a") as log:
        # its own process group on Windows, so a ctrl-break reaches it alone
        group = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else {}
        return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **group)


def git(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=timeout,
                          env=os.environ | dict(GIT_TERMINAL_PROMPT="0"))


def sync_results(folder: str) -> str:
    """Commit an experiment folder alone and push it: what happened, for the page."""
    path, name = str(Path(folder).resolve()), os.path.basename(folder)
    steps = [("add", "--", path), ("commit", "-m", f"data: {name}", "--", path),
             ("pull", "--rebase", "--autostash"), ("push",)]
    try:
        for args in steps:
            done = git(*args, timeout=600)
            if done.returncode:
                reason = (done.stderr or done.stdout).strip().splitlines() or [f"exit {done.returncode}"]
                return f"sync failed (git {args[0]}): {reason[-1]}"
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"sync failed: {error}"
    return f"{name} synced"


# a force pull keeps local commits on a branch and uncommitted changes in the stash
BACKUP_BRANCH = "pre-force-pull"
FORCE_PULL = (("branch", "-f", BACKUP_BRANCH), ("stash",), ("reset", "--hard", "@{u}"))
FORCE_PUSHED = (*FORCE_PULL[:-1], ("reset", "--hard", "FETCH_HEAD"))  # the host, to the commits sent it
FORCE_PULL_NOTE = f"Local commits stay on the branch {BACKUP_BRANCH}, uncommitted changes in git stash."
PULL_REFUSED = 75  # the host's checkout can't fast-forward: its exit status, so the page asks to force pull


def last_line(done: subprocess.CompletedProcess) -> str:
    return ((done.stderr or done.stdout).strip().splitlines() or [f"exit {done.returncode}"])[-1]


def ask_force_pull(reason: str) -> bool:
    if not sys.stdin.isatty():
        return False
    print(f"This checkout can't fast-forward to origin ({reason}).\n{FORCE_PULL_NOTE}")
    return input("Force pull, resetting it to origin? [y/N] ").strip().lower() in ("y", "yes")


def update_checkout() -> tuple[str | None, bool, str | None]:
    """Pull this checkout, offering a force pull if it diverged: its remote's url, whether new commits came in,
    and why the pull failed, if it did."""
    try:
        before = git("rev-parse", "HEAD").stdout
        remote = git("remote", "get-url", "origin").stdout.strip() or None
        if (fetched := git("fetch")).returncode:
            return remote, False, f"git fetch failed: {last_line(fetched)}"
        if (merged := git("merge", "--ff-only", "@{u}")).returncode:
            if not ask_force_pull(last_line(merged)):
                return remote, False, f"git pull failed: {last_line(merged)}"
            for args in FORCE_PULL:
                if (done := git(*args)).returncode:
                    return remote, False, f"force pull failed (git {args[0]}): {last_line(done)}"
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, False, f"git pull failed: {error}"
    return remote, git("rev-parse", "HEAD").stdout != before, None


def send_commits(tunnel: Tunnel, repo: str) -> tuple[str | None, str | None]:
    """Upload the commits the host's checkout lacks from this one as a git bundle, so the host needs no internet:
    the bundle's path there (None if it lacks nothing), and why sending failed, if it did."""
    try:
        theirs = tunnel.execute(f"git -C {shlex.quote(repo)} rev-parse --verify -q HEAD").strip()
        ours = git("rev-parse", "HEAD").stdout.strip()
        known = bool(theirs) and git("cat-file", "-e", f"{theirs}^{{commit}}").returncode == 0
        if theirs == ours or (known and git("merge-base", "--is-ancestor", "HEAD", theirs).returncode == 0):
            return None, None
        # only what the host lacks when this checkout knows its HEAD, else everything (a clone)
        branch = git("symbolic-ref", "-q", "HEAD").stdout.strip()
        revs = [f"{theirs}..HEAD"] if known else ["HEAD", *([branch] if branch else [])]
        fd, local = tempfile.mkstemp(suffix=".bundle")
        os.close(fd)
        try:
            if (bundled := git("bundle", "create", local, *revs)).returncode:
                return None, f"git bundle failed: {last_line(bundled)}"
            return tunnel.upload(local, "exopt.bundle"), None
        finally:
            os.remove(local)
    except (OSError, subprocess.TimeoutExpired, paramiko.SSHException) as error:
        return None, f"sending commits failed: {error}"


def controller_command(repo: str, bundle: str | None, remote: str | None, script: str, pull: str = "pull",
                       warning: str | None = None) -> str:
    """Shell command bringing the host's checkout to this one's commits from a bundle (cloning it if missing),
    then running a controller script in it.

    pull: "pull" exits with PULL_REFUSED if it can't fast-forward, "force" resets it, "skip" runs it as is."""
    folder = shlex.quote(repo)
    in_repo = lambda *args: f"git -C {folder} {shlex.join(args)}"
    clone = f"echo {shlex.quote(f'no commits to clone exopt from: {warning}')}; false"
    if bundle:
        set_origin = f" && {in_repo('remote', 'set-url', 'origin', remote)}" if remote else ""
        clone = f"git clone {shlex.quote(bundle)} {folder}{set_origin}"
    update = {
        "pull": f"{in_repo('merge', '--ff-only', 'FETCH_HEAD')} || exit {PULL_REFUSED}",
        "force": " && ".join(in_repo(*args) for args in FORCE_PUSHED)
                 + " || echo 'warning: force update failed, running the checkout as it is'",
        "skip": "echo 'running the checkout as it is'",
    }[pull]
    fetch = (f"if {in_repo('fetch', '-q', bundle, 'HEAD')}; then {update}; "
             f"else echo 'warning: git fetch from the bundle failed, running the checkout as it is'; fi"
             if bundle and pull != "skip" else "true")
    return (f'export GIT_TERMINAL_PROMPT=0 PATH="$HOME/.local/bin:$PATH"; '
            + (f"echo {shlex.quote(f'warning: {warning}, running the checkout as it is')}; " if warning else "")
            + f"if [ -d {folder}/.git ]; then {fetch}; else {clone} || exit 1; fi; "
            # no project build: it needs hatchling from PyPI, and the host may be offline
            f"cd {folder} && uv sync -q --frozen --inexact --no-install-project --group controller && "
            f"PYTHONPATH=src exec uv run --no-sync python {shlex.quote(script)}")


def run_dashboard(exp_dir: str, host: str, port: int, exo: tuple[str, str, int], atoms: int, script: str,
                  remote: str | None, warning: str | None):
    from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

    # drain the controller's sample stream from a background thread: a rolling buffer, and a thinned whole record
    stream, _ = link.driver_link(host)
    buffer: deque[dict] = deque(maxlen=int(BUFFER_SECONDS / 0.01))
    trajectory: deque[tuple] = deque(maxlen=int(4 * 3600 / 0.01 / TRAJECTORY_STRIDE))
    received = dict(at=-math.inf)

    def pump():
        count = 0
        while True:
            sample = cast(dict, stream.recv_json())
            # a controller restart resets its clock: drop the previous run
            if buffer and sample["time"] < buffer[-1]["time"]:
                buffer.clear()
                trajectory.clear()
            buffer.append(sample)
            if count % TRAJECTORY_STRIDE == 0:
                trajectory.append(tuple(math.nan if sample[k] is None else float(sample[k]) for k in TRAJECTORY_KEYS))
            count += 1
            received["at"] = time.monotonic()

    threading.Thread(target=pump, daemon=True).start()

    # the folder and surrogate shown, and the bo-loop launched from the page, if any
    experiments = os.path.dirname(exp_dir) or "."
    run = dict(exp_dir=exp_dir, process=None, started=None, t0=None)
    # the session shown (None: the landing page; name None: review only) and the configure window's state
    state: dict = dict(session=None, tunnel=None, repo=None, autostart=False)
    start_lock = threading.Lock()
    repo_name = posixpath.basename(remote or "exopt").removesuffix(".git")
    wizard: dict = dict(step=0, tunnel=None, host=None, fresh=False, editing=None)

    tile = lambda label, value, sub: html.Div(className="tile", children=[
        html.Div(label, className="label"), html.Div(value, className="value"), html.Div(sub, className="sub"),
    ])
    graph = lambda id: html.Div(className="card", children=dcc.Graph(id=id, config=dict(displaylogo=False)))
    grid = lambda *ids: html.Div(className="grid", children=[graph(id) for id in ids])
    app = Dash(__name__, title="exopt", update_title=None)
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + PAGE_CSS + "</style>")
    labeled = lambda label, field: html.Span([label, " ", field])
    field = lambda label, control, help="": html.Div(className="field", children=[
        html.Label(label), control, html.Div(help, className="help") if help else None])
    number = lambda id, value, width, **kw: dcc.Input(id=id, type="number", value=value, style=dict(width=width), **kw)
    defaults = trials.DEFAULT._replace(atoms=atoms)

    def setting_field(name: str):
        label, hint = trials.SETTINGS_HELP[name]
        default = getattr(defaults, name)
        step = 1 if isinstance(default, int) else "any"
        return field(label, number(f"exp-{name}", default, "100%", min=0, step=step), hint or "")
    exo_host, exo_user, exo_port = exo
    dropdown = lambda id, width, **kw: dcc.Dropdown(id=id, clearable=False, style=dict(
        width=width, display="inline-block", verticalAlign="middle"), **kw)
    hide = dict(display="none")
    landing = html.Div(id="landing", className="landing", children=html.Div(className="card", children=[
        html.H3("Connect to the exoskeleton"),
        html.P("Pick a controller and connect. To add one, choose “+ configure new” in the list."),
        dropdown("controller-pick", "100%", placeholder="configure a controller"),
        html.Div(className="controls", style=dict(marginTop="16px"), children=[
            html.Button("Connect", id="connect", className="primary"),
            html.Button("Edit", id="edit-controller", style=hide), html.Button("Delete", id="delete-controller", style=hide),
            html.Span(id="landing-note"),
        ]),
        html.Div(className="controls", children=html.Button("Review finished experiments", id="review-only")),
        dcc.ConfirmDialog(id="delete-confirm"),
    ]))
    configure_new = html.Div(id="wizard", className="modal", style=hide, children=[
        html.Div(className="card", children=[
            html.H3("Configure a controller", id="wizard-title"),
            html.Div(id="wizard-ssh", children=[
                html.P("1 / 3 · ssh host: the machine running the controller."),
                dcc.RadioItems(id="add-source", value="config", inline=True, options=[
                    dict(label=" from ssh config  ", value="config"), dict(label=" manually", value="manual")]),
                html.Div(id="from-config", className="controls", children=[
                    labeled("host", dropdown("config-alias", "320px", placeholder="a host of ~/.ssh/config")),
                ]),
                # saved to ~/.ssh/config; an existing name gets the new address (the Pi's changes with DHCP)
                html.Div(id="from-manual", className="controls", children=[
                    dcc.Input(id="new-name", placeholder="name", style=dict(width="120px")),
                    labeled("address", dcc.Input(id="new-address", value=exo_host, style=dict(width="160px"))),
                    labeled("user", dcc.Input(id="new-user", value=exo_user, style=dict(width="90px"))),
                    labeled("port", number("new-port", exo_port, "80px", min=1)),
                ]),
                # written to the host's ssh config entry as Password; blank keeps the current one, or ssh keys
                html.Div(className="controls", children=[
                    labeled("password", dcc.Input(id="new-password", type="password",
                                                  placeholder="blank: ssh key or unchanged", style=dict(width="220px"))),
                ]),
            ]),
            html.Div(id="wizard-repo", style=hide, children=[
                html.P("2 / 3 · where exopt lives on the host: a checkout there is pulled on each connect, "
                       "else it is cloned there."),
                html.Div(className="controls", children=[
                    html.Button("↑", id="browse-up", title="parent folder"),
                    html.Span(id="browse-path", className="path"),
                ]),
                html.Div(id="browse-list", className="browser"),
                html.Div(className="controls", children=html.Span(id="repo-label")),
            ]),
            html.Div(id="wizard-name", style=hide, children=[
                html.P("3 / 3 · nickname, under which this host and folder are saved."),
                html.Div(className="controls", children=dcc.Input(id="nickname", style=dict(width="220px"))),
            ]),
            html.Div(className="controls", style=dict(marginTop="12px"), children=[
                html.Button("Back", id="wizard-back"), html.Button("Next", id="wizard-next", className="primary"),
                html.Button("Cancel", id="wizard-cancel"), html.Span(id="wizard-note"),
            ]),
        ]),
    ])
    config = html.Div(id="config", className="modal", style=hide, children=[
        html.Div(className="card", children=[
            html.H3("New experiment"),
            html.P("Each experiment gets its own folder. The defaults are fine for a normal session."),
            html.Div(className="form", children=[
                field("Name", dcc.Input(id="exp-name", placeholder="e.g. subject1", style=dict(width="100%")),
                      html.Span(id="exp-folder")),
                *map(setting_field, BASIC_SETTINGS),
            ]),
            html.Details(className="advanced", children=[
                html.Summary("Advanced"),
                html.Div(className="form", children=[setting_field(name) for name in trials.Settings._fields
                                                     if name not in BASIC_SETTINGS]),
            ]),
            html.Div(className="controls", style=dict(marginTop="20px"), children=[
                html.Button("Start", id="start", className="primary"), html.Button("Cancel", id="cancel"),
                html.Span(id="exp-note", className="error-text"),
            ]),
        ]),
    ])
    # the controller's terminal on the host, streamed live beside the status
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
                html.Button("Restart controller", id="controller-start", className="primary", style=hide),
                html.Button("Reconnect", id="reconnect", className="primary", style=hide),
                html.Button("Stop experiment", id="stop", className="danger", style=hide),
                html.Span(id="sync-note", className="hint"),
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
    ])
    app.layout = html.Div(className="page", children=[
        html.Header(className="topbar", children=[
            html.Span("exopt", className="brand"),
            html.Span(id="session-pill", className="pill", style=hide),
            html.Span(id="experiment-pill", className="pill", style=hide),
            html.Span(warning, className="pill warn") if warning else None,
            html.Span(className="spacer"),
            html.Button("Disconnect", id="disconnect", style=hide),
        ]),
        landing,
        html.Div(id="session", style=hide, children=[status_section, live_section, review_section]),
        config, configure_new,
        dcc.Store(id="selected"), dcc.Store(id="latest-trial"),
        dcc.Store(id="theme"),
        dcc.ConfirmDialog(id="sync-confirm", message="Do you want to sync the results?"),
        dcc.ConfirmDialog(id="force-pull-confirm"),
        dcc.Store(id="live-loop"),
        dcc.Store(id="browse-dir"), dcc.Store(id="repo"), dcc.Store(id="ssh-checked"),
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
        current = trials.load_current(run["exp_dir"])
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
                    profile = trials.decode_profile(current) if current else None
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
        settings = trials.load_settings(run["exp_dir"])
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

    def picker_options() -> list[dict]:
        known = known_hosts()
        options = []
        for c in saved_controllers():
            h = known.get(c["host"])
            where = f"{h['user']}@{h['hostname']}" if h else f"host {c['host']} missing from {SSH_CONFIG}"
            options.append(dict(label=f"{c['name']} · {where} · {c['repo']}", value=c["name"]))
        return [*options, dict(label="+ configure new", value=NEW)]

    def close_wizard_tunnel():
        if wizard["tunnel"] is not None:
            wizard["tunnel"].close()
        wizard["tunnel"] = None

    @app.callback(Output("ssh-checked", "data"), Input("fast", "n_intervals"))
    def ssh_checked(_):
        """The configure window's ssh check finishing, apart from its clicks so a poll never overtakes one."""
        checking = cast(Tunnel | None, wizard["tunnel"])
        done = wizard["step"] == 1 and checking is not None and (checking.connected or checking.error)
        return time.monotonic() if done else no_update

    @app.callback(
        Output("controller-pick", "options"), Output("controller-pick", "value"), Output("wizard", "style"),
        Output("wizard-ssh", "style"), Output("wizard-repo", "style"), Output("wizard-name", "style"),
        Output("wizard-back", "style"), Output("wizard-next", "children"), Output("wizard-note", "children"),
        Output("nickname", "value"), Output("new-password", "value"), Output("config-alias", "options"),
        Output("config-alias", "value"), Output("add-source", "value"), Output("wizard-title", "children"),
        Input("controller-pick", "value"), Input("wizard-next", "n_clicks"), Input("wizard-back", "n_clicks"),
        Input("wizard-cancel", "n_clicks"), Input("ssh-checked", "data"), Input("edit-controller", "n_clicks"),
        Input("delete-confirm", "submit_n_clicks"),
        State("add-source", "value"), State("config-alias", "value"), State("new-name", "value"),
        State("new-address", "value"), State("new-user", "value"), State("new-port", "value"),
        State("new-password", "value"), State("repo", "data"), State("nickname", "value"),
    )
    def configure(picked, _next, _back, _cancel, _, _edit, _delete, source, config_alias, name, address, user, port,
                  password, repo, nickname):
        """The configure window: 1 ssh host (connected to check it), 2 exopt folder, 3 nickname, then saved."""
        trigger, step, note = ctx.triggered_id, wizard["step"], ""
        value = nick = password_field = alias_field = source_field = no_update
        connecting = cast(Tunnel | None, wizard["tunnel"])
        last_used = next((c["name"] for c in saved_controllers()), None)
        if trigger is None:
            value = last_used
        elif trigger == "ssh-checked":
            if step != 1 or connecting is None or not (connecting.connected or connecting.error):
                return (no_update,) * 15
            if connecting.connected:
                wizard.update(step=2, fresh=True)
            else:
                note = f"ssh to {connecting.target} failed: {connecting.error}"
                wizard["tunnel"] = None
        elif trigger == "controller-pick" and picked == NEW:
            close_wizard_tunnel()
            wizard.update(step=1, host=None, editing=None)
            value, password_field = last_used, ""
        elif trigger == "edit-controller" and picked not in (None, NEW):
            close_wizard_tunnel()
            edited = next(c for c in saved_controllers() if c["name"] == picked)
            wizard.update(step=1, host=None, editing=picked)
            password_field, alias_field, source_field = "", edited["host"], "config"
        elif trigger == "delete-confirm" and picked not in (None, NEW):
            delete_controller(picked)
            value = next((c["name"] for c in saved_controllers()), None)
        elif trigger == "wizard-cancel":
            close_wizard_tunnel()
            wizard.update(step=0, editing=None)
        elif trigger == "wizard-back" and step > 1:
            if step == 2:
                close_wizard_tunnel()
            wizard["step"] = step - 1
        elif trigger == "wizard-next" and step == 1 and connecting is None:
            alias = config_alias if source == "config" else (name or "").strip()
            if source == "config" and not alias:
                note = "pick a host"
            elif source == "manual" and not (alias and address and address.strip() and user and user.strip() and port):
                note = "name, address, user and port are needed"
            elif source == "manual" and any(c in alias for c in " *?!"):
                note = "the name cannot hold spaces or * ? !"
            else:
                try:
                    if source == "manual":
                        save_host(alias, address.strip(), user.strip(), int(port), password or None)
                    elif password:
                        save_host(alias, password=password)
                    h = known_hosts()[alias]
                except (OSError, KeyError) as error:
                    note = f"could not write {SSH_CONFIG}: {error}"
                else:
                    connecting = Tunnel(h["hostname"], h["user"], h["port"], h["password"], (), tuple(h["keys"]))
                    wizard.update(host=alias, tunnel=connecting)
                    note = f"connecting to {connecting.target}…"
        elif trigger == "wizard-next" and step == 2:
            if repo:
                wizard["step"], nick = 3, wizard["editing"] or wizard["host"]
            else:
                note = "open the folder exopt lives in"
        elif trigger == "wizard-next" and step == 3:
            nickname = (nickname or "").strip()
            if not nickname or nickname.startswith("::"):
                note = "a nickname is needed"
            elif not repo:
                note = "no folder for exopt: go back to pick it"
            else:
                if wizard["editing"] not in (None, nickname):
                    delete_controller(wizard["editing"])  # renamed
                save_controller(nickname, wizard["host"], repo)
                close_wizard_tunnel()
                wizard.update(step=0, editing=None)
                value = nickname

        step = wizard["step"]
        shown = lambda s: dict() if step == s else hide
        others = [dict(label=f"{a} · {h['user']}@{h['hostname']}", value=a) for a, h in known_hosts().items()]
        return (picker_options(), value, dict(display="flex") if step else hide, shown(1), shown(2), shown(3),
                dict() if step > 1 else hide, "Save" if step == 3 else "Next", note, nick, password_field, others,
                alias_field, source_field, f"Edit {wizard['editing']}" if wizard["editing"] else "Configure a controller")

    @app.callback(
        Output("edit-controller", "style"), Output("delete-controller", "style"), Input("controller-pick", "value"),
    )
    def picked_actions(picked):
        """Edit and delete only for a configured controller."""
        return (hide, hide) if picked in (None, NEW) else (dict(), dict())

    @app.callback(
        Output("delete-confirm", "displayed"), Output("delete-confirm", "message"),
        Input("delete-controller", "n_clicks"), State("controller-pick", "value"), prevent_initial_call=True,
    )
    def confirm_delete(_, picked):
        return True, f"Forget the controller {picked}? Its ssh host and the host's checkout stay."

    @app.callback(Output("from-config", "style"), Output("from-manual", "style"), Input("add-source", "value"))
    def add_source(source):
        return (dict(), hide) if source == "config" else (hide, dict())

    @app.callback(
        Output("browse-dir", "data"), Output("browse-list", "children"), Output("browse-path", "children"),
        Output("repo", "data"), Output("repo-label", "children"),
        Input("wizard-repo", "style"), Input("browse-up", "n_clicks"), Input(dict(entry=ALL), "n_clicks"),
        State("browse-dir", "data"),
    )
    def browse(card, _up, _entries, folder):
        """The host's folders over sftp: the one open holds exopt's checkout, or gets it cloned in."""
        current = cast(Tunnel | None, wizard["tunnel"])
        trigger = ctx.triggered_id
        if card.get("display") == "none" or current is None or not current.connected:
            return (no_update,) * 5
        if wizard["fresh"]:
            wizard["fresh"], folder = False, None
        # a listing exists already: only clicks change it (freshly listed entries fire with no clicks)
        elif trigger in (None, "wizard-repo") or not ctx.triggered[0]["value"]:
            return (no_update,) * 5

        # start in the edited controller's checkout, else one on the same host, else the home folder
        if folder is None:
            saved = saved_controllers()
            known = [c["repo"] for c in saved if c["host"] == wizard["host"] and c["name"] == wizard["editing"]] + \
                    [c["repo"] for c in saved if c["host"] == wizard["host"]]
            target = known[0] if known else "."
        elif trigger == "browse-up":
            target = posixpath.dirname(folder.rstrip("/")) or "/"
        else:
            target = posixpath.join(folder, trigger["entry"])
        try:
            folder, entries = current.browse(target)
        except OSError as error:
            try:
                if folder is not None or target == ".":
                    raise
                folder, entries = current.browse(".")  # that folder is gone
            except OSError:
                return no_update, no_update, f"{target}: {error}", no_update, no_update

        # the open folder if it is a checkout, else exopt's folder in it
        checkout = lambda path: current.exists(posixpath.join(path, ".git"))
        repo = folder if checkout(folder) else posixpath.join(folder, repo_name)
        label = f"{repo}: pulled on each connect" if checkout(repo) else f"{repo}: cloned on connect"
        rows = [html.Button(name + "/", id=dict(entry=name)) for name, is_folder in entries if is_folder]
        return folder, rows or [html.Span("no folders", className="path")], folder, repo, label

    def experiment_running() -> bool:
        return run["process"] is not None and run["process"].poll() is None

    def open_session(name: str) -> str:
        """Open a session (a configured controller's: its tunnel, then its controller); why not, if it cannot."""
        config = next((c for c in saved_controllers() if c["name"] == name), None)
        found = known_hosts().get(config["host"]) if config else None
        if config is None or found is None:
            return f"{name}: its ssh host is no longer in {SSH_CONFIG}, configure it again"
        save_controller(config["name"], config["host"], config["repo"])  # last used first
        buffer.clear()
        trajectory.clear()
        opened = Tunnel(found["hostname"], found["user"], found["port"], found["password"],
                        (link.SAMPLE_PORT, link.PROFILE_PORT), tuple(found["keys"]))
        state.update(session=dict(name=name), tunnel=opened, repo=config["repo"], autostart=True)
        return ""

    def close_session():
        if state["tunnel"] is not None:
            state["tunnel"].close()  # ctrl-c to the controller first
        state.update(session=None, tunnel=None, repo=None, autostart=False)

    @app.callback(
        Output("landing", "style"), Output("session", "style"), Output("disconnect", "style"),
        Output("session-pill", "style"), Output("experiment-pill", "style"), Output("landing-note", "children"),
        Output("status", "style"), Output("live-section", "style"), Output("disconnect", "children"),
        Input("connect", "n_clicks"), Input("review-only", "n_clicks"), Input("disconnect", "n_clicks"),
        Input("fast", "n_intervals"), State("controller-pick", "value"),
    )
    def session(_connect, _review, _disconnect, _, picked):
        """The landing page, or the session (a reloaded page returns to the open one)."""
        note = no_update if ctx.triggered_id == "fast" else ""
        if ctx.triggered_id == "connect" and state["session"] is None:
            note = open_session(picked) if picked not in (None, NEW) else "configure a controller first"
        elif ctx.triggered_id == "review-only" and state["session"] is None:
            state.update(session=dict(name=None))
        elif ctx.triggered_id == "disconnect" and not experiment_running():
            close_session()
        if state["session"] is None:
            return dict(), hide, hide, hide, hide, note, no_update, no_update, no_update
        if state["session"]["name"] is None:  # review only: no controller, no live view
            return hide, dict(), dict(), hide, hide, note, hide, hide, "Back"
        return hide, dict(), dict(), dict(), dict(), note, dict(), dict(), "Disconnect"

    @app.callback(
        Output("config", "style"), Output("exp-note", "children"),
        Input("configure", "n_clicks"), Input("start", "n_clicks"), Input("cancel", "n_clicks"),
        State("exp-name", "value"), prevent_initial_call=True,
    )
    def config_window(_configure, _start, _cancel, name):
        if ctx.triggered_id == "configure":
            return dict(display="flex"), ""
        if ctx.triggered_id == "start" and not (name and name.strip()):
            return no_update, "give the experiment a name"
        return hide, ""

    @app.callback(Output("exp-folder", "children"), Input("exp-name", "value"))
    def experiment_folder(name):
        slug = (name or "").strip().replace(" ", "-") or "<name>"
        return f"saved to {experiments}/{time.strftime('%Y-%m-%d')}-{slug}"

    @app.callback(
        Output("sync-confirm", "displayed"),
        Input("stop", "n_clicks"), Input("sync-confirm", "submit_n_clicks"), prevent_initial_call=True,
    )
    def sync_prompt(_stop, _submit):
        process, folder = run["process"], run["exp_dir"]
        if ctx.triggered_id == "stop":
            return experiment_running()
        # commit once bo-loop has ramped down and written its last trial
        def sync():
            run["sync"] = f"syncing {os.path.basename(folder)}…"
            if process is not None:
                process.wait()
            run["sync"] = sync_results(folder)
        threading.Thread(target=sync, daemon=True).start()
        return False

    @app.callback(
        Output("force-pull-confirm", "displayed"), Output("force-pull-confirm", "message"),
        Input("slow", "n_intervals"), prevent_initial_call=True,
    )
    def force_pull_prompt(_):
        """Ask once per refused pull whether to force pull the host's checkout."""
        tunnel = cast(Tunnel | None, state["tunnel"])
        command = tunnel.command if tunnel is not None else None
        if command is None or command.exit_status() != PULL_REFUSED or state.get("pull_asked") is command:
            return no_update, no_update
        state["pull_asked"] = command
        reason = (command.output.strip().split("\n") or [""])[-1]
        return True, (f"{state['session']['name']}'s checkout {state['repo']} can't fast-forward to this computer's"
                      f"{f' ({reason})' if reason else ''}.\n\nForce update, resetting it to this computer's? "
                      f"{FORCE_PULL_NOTE}"
                      "\nCancel runs the checkout as it is.")

    @app.callback(
        Output("force-pull-confirm", "displayed", allow_duplicate=True),
        Input("force-pull-confirm", "submit_n_clicks"), Input("force-pull-confirm", "cancel_n_clicks"),
        prevent_initial_call=True,
    )
    def force_pull_answer(_submit, _cancel):
        tunnel = cast(Tunnel | None, state["tunnel"])
        if tunnel is None or not (tunnel.connected and tunnel.alive()):
            return False
        pull = "force" if ctx.triggered[0]["prop_id"].endswith("submit_n_clicks") else "skip"
        with start_lock:
            tunnel.start(controller_command(state["repo"], state.get("bundle"), remote, script, pull))
        return False

    def session_status(last: dict | None, current: dict | None, history: list[trials.Trial], segment: dict,
                       window: dict, settings: trials.Settings) -> Status:
        """Where the session is, from the tunnel, the controller, its stream and the experiment."""
        tunnel = cast(Tunnel, state["tunnel"])
        name = state["session"]["name"]
        command = tunnel.command if tunnel.alive() else None
        if not tunnel.alive():
            return Status(0, "error", f"Can't reach {name}", tunnel.error or "The ssh connection dropped.", "reconnect")
        if not tunnel.connected:
            return Status(0, "busy", f"Connecting to {name}…", tunnel.target)
        if last is None:
            if command is not None and command.running():
                if received["at"] > -math.inf:
                    return Status(0, "error", "No data from the controller",
                                  "It stopped streaming. Check its output below.", "controller-start")
                return Status(0, "busy", "Starting the controller…", "This takes a few seconds.")
            output = (command.output.strip().split("\n") or [""])[-1] if command is not None else ""
            return Status(0, "error", "The controller stopped", output or "Restart it to continue.",
                          "controller-start")
        if experiment_running():
            return trial_status(last, current, history, segment, settings)
        recent = window["time"] > window["time"][-1] - 2.0 if window else []
        moving = window and any(np.nanmax(np.abs(window[f"velocity_{leg}"][recent])) > MOVING_SPEED for leg in (0, 1))
        if last["stage"] == "calibrating" or (last["stage"] == "standing" and moving):
            return Status(1, "busy", "Keep walking…", "Calibrating the gait: it takes a few strides. The torque "
                          "stays off until then.")
        if last["stage"] == "standing":
            return Status(1, "action", "Start walking", "The gait is calibrated after a few strides; the torque "
                          "stays off until then.")
        process = run["process"]
        ended = f" The last experiment stopped after {len(history)} trials." if process is not None and history else ""
        return Status(2, "ready", "Ready to start an experiment",
                      "The subject keeps walking while profiles are tried, one after the other." + ended, "configure")

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

    actions = ("configure", "controller-start", "reconnect", "stop")

    @app.callback(
        Output("status", "className"), Output("stepper", "children"), Output("hero-eyebrow", "children"),
        Output("hero-title", "children"), Output("hero-detail", "children"), Output("hero-progress", "children"),
        *(Output(id, "style") for id in actions),
        Output("controller-log", "children"), Output("console-summary", "children"),
        Output("controller-stop", "disabled"), Output("tiles", "children"),
        Output("session-pill", "children"), Output("experiment-pill", "children"), Output("disconnect", "disabled"),
        Output("sync-note", "children"),
        Input("controller-start", "n_clicks"), Input("controller-stop", "n_clicks"),
        Input("start", "n_clicks"), Input("stop", "n_clicks"), Input("reconnect", "n_clicks"),
        Input("fast", "n_intervals"),
        State("exp-name", "value"), *(State(f"exp-{name}", "value") for name in trials.Settings._fields),
    )
    def status(*args):
        """Act on a click, then render the session from server state (so a dropped response heals on the next tick)."""
        name, values = args[-1 - len(trials.Settings._fields)], args[-len(trials.Settings._fields):]
        if state["session"] is None or state["session"]["name"] is None:
            return (no_update,) * 18
        tunnel, trigger = cast(Tunnel, state["tunnel"]), ctx.triggered_id

        # clicks: reconnect, the controller, the experiment
        if trigger == "reconnect" and not experiment_running():
            controller_name = state["session"]["name"]
            close_session()
            open_session(controller_name)
            tunnel = cast(Tunnel, state["tunnel"])
        connected = tunnel.connected and tunnel.alive()
        # ticks run concurrently: without the lock two of them autostart two controllers (racing clones)
        with start_lock:
            command = tunnel.command
            running = command is not None and command.running()
            if (trigger == "controller-start" or (state["autostart"] and command is None)) and connected:
                if running:
                    command.interrupt()
                state["bundle"], failed = send_commits(tunnel, state["repo"])
                command = tunnel.start(controller_command(state["repo"], state["bundle"], remote, script,
                                                          warning=failed))
                running, state["autostart"] = True, False
        if trigger == "controller-stop" and running:
            command.interrupt()
        elif trigger == "start" and not experiment_running() and name and name.strip():
            # a cleared field keeps its default; bo-loop clamps the gait counts
            options = {k: default if v is None else type(default)(v)
                       for (k, default), v in zip(defaults._asdict().items(), values)}
            folder = os.path.join(experiments, f"{time.strftime('%Y-%m-%d')}-{name.strip().replace(' ', '-')}")
            run.update(process=launch_bo_loop(folder, host, options), started=time.monotonic(), exp_dir=folder, sync=None,
                       t0=buffer[-1]["time"] if buffer else None)
        elif trigger == "stop" and experiment_running():
            # an interrupt, so bo-loop releases the profile and the torque ramps down
            run["process"].send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)

        # what is going on
        data = list(buffer)
        last = data[-1] if data and time.monotonic() - received["at"] < STALE_AFTER else None
        window = columns([x for x in data if x["time"] > data[-1]["time"] - LIVE_WINDOW]) if data else {}
        current = trials.load_current(run["exp_dir"]) if experiment_running() else None
        history, settings = load_trials(run["exp_dir"]), trials.load_settings(run["exp_dir"])
        segment = columns([x for x in data if current and x["profile_id"] == current["id"] and x["gain"] >= 1]
                          [settings.warmup:])
        now = session_status(last, current, history, segment, window, settings)

        # the steps, the headline and its one action
        steps = []
        for k, step_name in enumerate(STEPS):
            kind = "done" if k < now.step else ("active" if k == now.step else "todo")
            steps.append(html.Li(className=f"step {kind}", children=[
                html.Span("✓" if kind == "done" else str(k + 1), className="num"), html.Span(step_name)]))
        shown = lambda id: dict() if id == now.action or (id == "stop" and experiment_running()) else hide

        # the controller's shell, folded away
        log = "\n".join(command.output.rstrip("\n").split("\n")[-200:]) if command is not None else ""
        if command is not None and not running:
            exit_status = command.exit_status()
            log += "\n· controller exited" + (f" ({exit_status})" if exit_status not in (None, -1) else "")
        summary = f"Controller terminal · live · {'running' if running else 'stopped'} · {posixpath.join(state['repo'], script)}"

        dot = {"error": "error", "busy": "busy"}.get(now.tone, "ok") if now.step or now.tone != "busy" else "busy"
        pill = [html.Span(className=f"dot {dot}"), html.B(state["session"]["name"]), tunnel.target]
        experiment = [html.Span("experiment"), html.B(os.path.basename(run["exp_dir"]))] \
            if run["process"] is not None else []
        return (
            f"status tone-{now.tone}", steps, now.eyebrow or f"step {now.step + 1} of {len(STEPS)}", now.title,
            now.detail, progress_bar(*now.progress) if now.progress else None,
            *(shown(id) for id in actions), log, summary, not running,
            [tile(*t) for t in live_tiles(window, last, segment, history, settings)], pill, experiment, experiment_running(),
            run.get("sync"),
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

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock", help="experiment folder")
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--exo-host", default="", help="manual new host's default address")
    parser.add_argument("--exo-user", default="pi", help="ssh user on the Pi")
    parser.add_argument("--exo-port", type=int, default=22, help="ssh port on the Pi")
    parser.add_argument("--atoms", type=int, default=trials.DEFAULT.atoms, help="new experiment default (as bo-loop)")
    parser.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)  # simulated legs on the host
    parser.add_argument("--diagnose", action="store_true", help="run the drive check instead of the controller")
    args = parser.parse_args()

    remote, updated, warning = update_checkout()
    # new commits: start over on them, with their dependencies
    if updated:
        try:
            sys.exit(subprocess.call(["uv", "run", "--project", str(REPO), "--group", "optimizer", "start", *sys.argv[1:]]))
        except OSError as error:
            warning = f"pulled new commits but could not restart on them ({error}): restart the dashboard"
    if warning:
        print(warning)
    run_dashboard(args.exp_dir, args.host, args.port, (args.exo_host, args.exo_user, args.exo_port), args.atoms,
                  DIAGNOSE if args.diagnose else MOCK_CONTROLLER if args.mock else CONTROLLER, remote, warning)


if __name__ == "__main__":
    main()
