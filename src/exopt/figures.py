"""Plotly figures of trials and live samples, and the palettes and page style they share."""

from typing import NamedTuple
from pathlib import Path
import math

import jax
import numpy as np
import plotly.graph_objects as go
from bofus.rkhs import X_MAX, RBFMixture
from einops import rearrange

from exopt import trials
from exopt.controller import MAX_TORQUE, PERIODIC, power_assist_torque, profile_torque, unit_phase

LIVE_WINDOW = 10.0  # [s] of leg traces in the running view
LIVE_STRIDE = 3  # plot every 3rd sample of the traces, so a refresh stays cheap
LIVE_HEIGHT = 680  # [px] of the four rows of leg traces
LIVE_ROWS = (("position", "rad"), ("velocity", "rad/s"), ("torque", "Nm"), ("phase", "cycles"))
POWER_BINS = 32
PHASE = np.linspace(0.0, 2 * math.pi, 241)
PHASE_CYCLES = PHASE / (2 * math.pi)  # unwrapped, so curves end at 1 instead of jumping back to 0
LEG_LENGTH = 1.0  # animation units
CLICKABLE = ("loss", "profiles")  # optimization figures, a click selects a trial
TRIAL = ("profile", "power", "convergence", "portrait")  # figures of the selected trial
WEB = Path(__file__).with_name("web")
EXPERIMENT_POINTS = 60_000  # samples at most in the whole experiment figure
GAP = 1.0  # [s] without samples that breaks the experiment's lines and spans
REGION_COLORS = {"metabolic": "rgba(60,170,90,0.20)", "ramp": "rgba(214,160,40,0.20)", "full torque, not scored": "rgba(137,135,129,0.18)",
                 "calibrating": "rgba(140,90,200,0.14)", "standing": "rgba(137,135,129,0.08)"}


class Palette(NamedTuple):
    """Reference palette, one step per mode; the loss ramp runs from best to worst."""

    surface: str
    page: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    border: str
    legs: tuple[str, str]
    accent: str
    band: str
    ramp: tuple[str, ...]


LIGHT = Palette(
    "#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "rgba(11,11,11,0.10)",
    ("#2a78d6", "#eb6834"), "#2a78d6", "rgba(42,120,214,0.15)",
    ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"),
)
DARK = Palette(
    "#1a1a19", "#0d0d0d", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", "rgba(255,255,255,0.10)",
    ("#3987e5", "#d95926"), "#3987e5", "rgba(57,135,229,0.22)",
    ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"),
)
THEMES = dict(light=LIGHT, dark=DARK)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def cycles(phase: np.ndarray) -> np.ndarray:
    """Gait phase [rad] as a fraction of the cycle in [0, 1)."""
    return np.mod(phase, 2 * math.pi) / (2 * math.pi)


def break_wraps(x: np.ndarray, y: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN gaps where the gait phase (in cycles) wraps, so a line does not jump back across the cycle."""
    at = np.flatnonzero(np.diff(phase) < -0.5) + 1
    return np.insert(x, at, np.nan), np.insert(y, at, np.nan)



def gait_period(s: dict[str, np.ndarray]) -> float | None:
    """Median time between phase wraps of leg 0."""
    ok = ~np.isnan(s["phase_0"])
    wraps = s["time"][ok][1:][np.diff(s["phase_0"][ok]) < -math.pi]
    return float(np.median(np.diff(wraps))) if len(wraps) >= 3 else None


def running_estimate(samples: dict[str, np.ndarray], settings: trials.Settings
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample index, loss and SE at the end of each valid gait from the second on, as trials.gait_estimate gives them."""
    power, gaits = trials.penalized_power(samples, settings.beta), trials.gait_slices(samples, settings)
    per_gait = np.array([-power[gait].mean() for gait in gaits])
    at, loss, se = [], [], []
    for k in range(2, len(gaits) + 1):
        at.append(gaits[k - 1].stop - 1)
        loss.append(-np.concatenate([power[gait] for gait in gaits[:k]]).mean())
        se.append(per_gait[:k].std(ddof=1) / np.sqrt(k))
    return np.array(at, int), np.array(loss), np.array(se)


def leg_power(samples: dict[str, np.ndarray], leg: int, beta: float) -> np.ndarray:
    p = samples[f"torque_{leg}"] * samples[f"velocity_{leg}"]
    return np.where(p < 0, beta * p, p)


def axis_style(p: Palette) -> dict:
    return dict(gridcolor=p.grid, linecolor=p.axis, zerolinecolor=p.axis, tickcolor=p.axis, title_font_size=12)


def layout_style(p: Palette, title: str | None = None, height: int = 300) -> dict:
    return dict(
        height=height, uirevision="keep", paper_bgcolor=p.surface, plot_bgcolor=p.surface,
        title=dict(text=title, x=0, font=dict(size=14, color=p.ink)) if title else None,
        font=dict(family=FONT, size=12, color=p.ink_2), margin=dict(l=56, r=16, t=44 if title else 16, b=44),
        hoverlabel=dict(bgcolor=p.surface, bordercolor=p.axis, font=dict(family=FONT, color=p.ink)),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"),
    )


def styled(p: Palette, fig: go.Figure, title: str | None = None, height: int = 300, **layout) -> go.Figure:
    fig.update_xaxes(**axis_style(p))
    fig.update_yaxes(**axis_style(p))
    return fig.update_layout(**layout_style(p, title, height), **layout)


def placeholder(p: Palette, message: str, title: str | None = None, height: int = 300) -> go.Figure:
    fig = go.Figure().add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color=p.muted, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return styled(p, fig, title, height)


def note(p: Palette, fig: go.Figure, text: str) -> go.Figure:
    """Right-aligned caption above the plot."""
    return fig.add_annotation(text=text, x=1, y=1, xref="paper", yref="paper", xanchor="right",
                              yanchor="bottom", showarrow=False, font=dict(color=p.ink_2))


def ring(p: Palette, fig: go.Figure, x: float, y: float) -> go.Figure:
    """Mark the selected trial."""
    return fig.add_scatter(x=[x], y=[y], mode="markers", name="selected", hoverinfo="skip",
                           marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=p.ink, width=2)))


def phase_axis(fig: go.Figure) -> go.Figure:
    return fig.update_xaxes(title_text="gait phase (cycles)", range=[0, 1])


def profile_figure(
    title: str,
    profile: RBFMixture | None,
    points: dict | None = None,
    label: str = "",
    atoms: bool = False,
    power_assist: float = 0.0,
    p: Palette = LIGHT,
) -> go.Figure:
    """A profile in Nm over the gait cycle, optionally with its atoms and per-leg torque samples."""
    if profile is None and not points:
        return placeholder(p, "no profile running", title)
    fig = go.Figure()
    for leg in (0, 1) if points else ():
        fig.add_scatter(x=cycles(points[f"phase_{leg}"]), y=points[f"{label}_{leg}"], mode="markers",
                        name=f"{label.replace('_', ' ')}, leg {leg}", hoverinfo="skip",
                        marker=dict(size=3.5, color=p.legs[leg], opacity=0.45))
    if profile is not None and atoms:
        # each atom's term in Nm before the tanh soft clip: they sum to the profile only where it is far from saturating
        atoms = jax.tree.map(lambda z: rearrange(z, "m ... -> m 1 ..."), profile)
        points = rearrange(unit_phase(PHASE), "p -> p 1")
        atom_torques = MAX_TORQUE * np.asarray(jax.vmap(lambda x: atoms(x, PERIODIC))(points)).T
        centers = (np.asarray(profile.x[:, 0]) / X_MAX + 1) / 2
        lengthscales = np.sqrt(np.asarray(profile.l[:, 0])) / (2 * X_MAX)
        for k, (atom_torque, c, ls, a) in enumerate(zip(atom_torques, centers, lengthscales, np.asarray(profile.a))):
            fig.add_scatter(x=PHASE_CYCLES, y=atom_torque, mode="lines", name="atoms (before tanh)",
                            legendgroup="atoms", showlegend=k == 0, line=dict(color=p.muted, width=1.5, dash="dash"),
                            hovertemplate=f"atom {k + 1}: center {c:.2f}, lengthscale {ls:.2f} cycles, "
                                          f"amplitude {a:+.2f}<extra></extra>")
    if profile is not None and power_assist:
        fig.add_scatter(x=PHASE_CYCLES, y=power_assist_torque(power_assist, PHASE), mode="lines", name="power assist",
                        line=dict(color=p.muted, width=1.5, dash="dot"), hoverinfo="skip")
    if profile is not None:
        fig.add_scatter(x=PHASE_CYCLES, y=profile_torque(profile, PHASE, power_assist), mode="lines",
                        name="profile", line=dict(color=p.ink, width=2),
                        hovertemplate="phase %{x:.2f}<br>%{y:.2f} Nm<extra></extra>")
    fig.update_yaxes(title_text="torque (Nm)")
    return styled(p, phase_axis(fig), title)


def estimate_figure(title: str, samples: dict, settings: trials.Settings, rule: dict | None = None, p: Palette = LIGHT
                    ) -> go.Figure | dict:
    """Loss estimate of a segment ± z SE at each valid gait, against the best observed profile's pessimistic mean.

    Built as a plain dict, cheap enough to redraw live."""
    at, loss, se = running_estimate(samples, settings) if len(samples.get("time", [])) > 1 else ([], [], [])
    if len(at) == 0:
        return placeholder(p, "waiting for 2 valid gaits", title)
    z = settings.stop_z
    t = (samples["time"][at] - samples["time"][0]).tolist()
    band = lambda low, high, color, name: dict(
        type="scatter", x=t + t[::-1], y=np.concatenate([high, low[::-1]]).tolist(), fill="toself", fillcolor=color,
        line=dict(width=0), name=name, hoverinfo="skip")
    line = lambda y, name, **style: dict(type="scatter", x=t, y=np.asarray(y).tolist(), name=name,
                                         **(dict(mode="lines") | style))
    data = []
    layout = layout_style(p, title) | dict(xaxis=axis_style(p) | dict(title=dict(text="time in segment (s)")),
                                           yaxis=axis_style(p) | dict(title=dict(text="loss")))
    if rule and "best_mean" in rule:
        best = np.full_like(loss, trials.pessimistic_best(rule, z))
        data.append(line(best, f"best (posterior mean + {z:g} sd)", line=dict(color=p.muted, width=1.5, dash="dash"),
                         hoverinfo="skip"))
    data += [band(loss - z * se, loss + z * se, p.band, f"segment ± {z:g} SE"),
             line(loss, "segment estimate", mode="lines+markers", line=dict(color=p.accent, width=2),
                  marker=dict(size=5), customdata=se.tolist(),
                  hovertemplate="%{x:.1f} s<br>loss %{y:.3f} ± %{customdata:.3f} SE<extra></extra>")]
    return dict(data=data, layout=layout)


def live_figure(s: dict[str, np.ndarray], p: Palette = LIGHT, stride: int = LIVE_STRIDE) -> go.Figure | dict:
    """Position, velocity, torque and estimated phase of both legs.

    Built as a plain dict: plotly's validation of a four-row figure is slow. Trace order is LIVE_ROWS by leg.
    """
    if not s:
        return placeholder(p, "no samples from the controller", height=LIVE_HEIGHT)
    rows = LIVE_ROWS
    t, gap = s["time"][::stride], 0.07
    height = (1 - gap * (len(rows) - 1)) / len(rows)
    data, layout = [], layout_style(p, height=LIVE_HEIGHT)
    for row, (kind, unit) in enumerate(rows, 1):
        suffix = "" if row == 1 else str(row)
        for leg in (0, 1):
            x, y = np.asarray(t), s[f"{kind}_{leg}"][::stride]
            if kind == "phase":
                x, y = break_wraps(x, cycles(y), cycles(y))
            data.append(dict(
                type="scatter", x=x.tolist(), y=y.tolist(), xaxis=f"x{suffix}", yaxis=f"y{suffix}", mode="lines",
                name=f"leg {leg}", legendgroup=f"leg {leg}", showlegend=row == 1, line=dict(color=p.legs[leg], width=1.5),
            ))
        top = 1 - (row - 1) * (height + gap)
        layout[f"xaxis{suffix}"] = axis_style(p) | dict(anchor=f"y{suffix}", showticklabels=row == len(rows))
        if row > 1:
            layout[f"xaxis{suffix}"]["matches"] = "x"
        layout[f"yaxis{suffix}"] = axis_style(p) | dict(
            anchor=f"x{suffix}", domain=[max(top - height, 0.0), top], title=dict(text=f"{kind} ({unit})"),
        )
    layout[f"xaxis{len(rows)}"]["title"] = dict(text="controller time (s)")
    return dict(data=data, layout=layout)


def experiment_regions(
    s: dict[str, np.ndarray], history: list[trials.Trial], events: list[tuple[int, dict]], experiment_time: np.ndarray
) -> list[tuple[str, float, float]]:
    """Spans of each sample's part of the experiment: a scored trial, a metabolic recording, a torque ramp, unscored
    full torque, a stage."""
    gain, stage = s["gain"], s["stage"]
    labels = np.where(stage != "running", stage, np.where((gain > 0) & (gain < 1), "ramp", np.where(
        gain >= 1, "full torque, not scored", None))).astype(object)
    for t in history:
        scored = (s["profile_id"] == t.samples["profile_id"][0]) & (s["time"] >= t.samples["time"][0]) & (
            s["time"] <= t.samples["time"][-1])
        labels[scored] = f"trial {t.index}"
    starts = [(row, e) for row, e in events if e.get("metabolic") == "start"]
    stops = [row for row, e in events if e.get("metabolic") == "stop"]
    for row, event in starts:
        stop = min((r for r in stops if r >= row), default=len(labels))
        labels[row:stop] = " ".join(["metabolic", event["condition"], event.get("label", "")]).strip()

    # runs of one label, broken where the recording paused
    regions, start = [], 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start] or experiment_time[k] - experiment_time[k - 1] > GAP:
            if labels[start] is not None:
                regions.append((labels[start], float(experiment_time[start]), float(experiment_time[k - 1])))
            start = k
    return regions


def experiment_figure(exp_dir: str, history: list[trials.Trial], p: Palette = LIGHT) -> dict:
    """Leg signals over the whole experiment, shaded by trial and transition; the trials alone before stream records."""
    s, events = trials.load_stream(exp_dir, EXPERIMENT_POINTS)
    recorded = bool(s)
    if not recorded:
        if not history:
            return placeholder(p, "no samples recorded", height=LIVE_HEIGHT).to_plotly_json()
        keys = [k for k in history[0].samples if k not in ("stage", "profile_id")]
        s = {k: np.concatenate([t.samples[k] for t in history]).astype(float if k in keys else object)
             for k in (*keys, "stage", "profile_id")}

    # one clock across controller restarts (each resets its time), the recording's pauses kept
    raw = s["time"]
    steps = np.diff(raw, prepend=raw[:1])
    experiment_time = raw - raw[0] + np.cumsum(np.where(steps < 0, -steps + 0.01, 0.0))
    regions = experiment_regions(s, history, events, experiment_time)
    at = np.flatnonzero(np.diff(experiment_time) > GAP) + 1
    numeric = {k: np.insert(v, at, np.nan) for k, v in s.items() if v.dtype != object} | dict(
        time=np.insert(experiment_time, at, np.nan))

    fig = live_figure(numeric, p, stride=1)
    for trace in fig["data"]:
        trace["type"] = "scattergl"
    for leg in (0, 1):
        fig["data"].append(dict(
            type="scattergl", x=numeric["time"].tolist(), y=numeric[f"measured_torque_{leg}"].tolist(), xaxis="x3",
            yaxis="y3", mode="lines", name=f"leg {leg} measured", line=dict(color=p.legs[leg], width=1, dash="dot"),
        ))

    # shaded spans across all rows, trials numbered on top, a legend entry per kind
    colors = REGION_COLORS | {"trial": p.band}
    kind = lambda label: next((k for k in ("trial", "metabolic") if label.startswith(k)), label)
    fig["layout"]["shapes"] = [
        dict(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=colors.get(kind(label), p.border),
             line=dict(width=0), layer="below") for label, x0, x1 in regions
    ]
    fig["layout"]["annotations"] = [
        dict(text=label.removeprefix("trial ").removeprefix("metabolic "), x=(x0 + x1) / 2, y=1, xref="x", yref="paper",
             yanchor="top", showarrow=False, font=dict(size=10, color=p.ink_2))
        for label, x0, x1 in regions if kind(label) != label
    ]
    for label in dict.fromkeys(map(kind, (label for label, *_ in regions))):
        fig["data"].append(dict(type="scatter", x=[None], y=[None], mode="markers", name=label,
                                marker=dict(symbol="square", size=12, color=colors.get(label, p.border))))
    fig["layout"]["xaxis4"]["title"] = dict(text="experiment time (s)" + ("" if recorded else ", scored trials only"))
    fig["layout"]["margin"]["t"] = 40
    return fig


def loss_figure(history: list[trials.Trial], selected: int | None, p: Palette = LIGHT) -> go.Figure:
    title = "loss per trial"
    if not history:
        return placeholder(p, "no trials yet", title, 320)
    index = np.array([t.index for t in history])
    loss = np.array([t.loss for t in history])
    se = np.array([t.se for t in history])
    best = int(loss.argmin())
    data = np.stack([index, se, [len(t.samples["time"]) for t in history]], 1)
    hover = "trial %{customdata[0]}<br>loss %{y:.3f} ± %{customdata[1]:.3f}<br>%{customdata[2]} samples<extra></extra>"

    fig = go.Figure()
    fig.add_scatter(x=index, y=np.minimum.accumulate(loss), mode="lines", name="best so far",
                    line=dict(color=p.accent, width=2, shape="hv"), hoverinfo="skip")
    fig.add_scatter(x=index, y=loss, mode="markers", name="trial", customdata=data.tolist(), hovertemplate=hover,
                    error_y=dict(array=2 * se, color=p.axis, thickness=1.5, width=0),
                    marker=dict(size=9, color=p.muted, line=dict(color=p.surface, width=2)))
    fig.add_scatter(x=index[[best]], y=loss[[best]], mode="markers", name=f"best (trial {index[best]})",
                    customdata=data[[best]].tolist(), hovertemplate=hover,
                    marker=dict(size=12, color=p.accent, line=dict(color=p.surface, width=2)))
    if selected in index:
        ring(p, fig, selected, loss[index == selected][0])
    fig.update_xaxes(title_text="trial", dtick=max(1, len(index) // 12))
    fig.update_yaxes(title_text="loss (±2 SE)")
    return styled(p, fig, title, 320)


def profiles_figure(history: list[trials.Trial], selected: int | None, p: Palette = LIGHT) -> go.Figure:
    title = "profiles tried"
    if not history:
        return placeholder(p, "no trials yet", title, 320)
    best = min(history, key=lambda t: t.loss)
    chosen = next((t for t in history if t.index == selected and t is not best), None)
    others = [t for t in history if t is not best and t is not chosen]

    # others faint, then the best, then the selected one on top
    fig = go.Figure()
    styles = [(t, "rgba(137,135,129,0.35)", 1, "other trials", k == 0) for k, t in enumerate(others)]
    styles += [(best, p.accent, 2.5, f"best (trial {best.index})", True)]
    styles += [(chosen, p.ink, 2, f"selected (trial {chosen.index})", True)] if chosen else []
    for t, color, width, name, legend in styles:
        applied = profile_torque(t.profile, PHASE, t.power_assist)
        fig.add_scatter(x=PHASE_CYCLES, y=applied, mode="lines", name=name,
                        showlegend=legend, line=dict(color=color, width=width), customdata=np.full(len(PHASE), t.index),
                        hovertemplate=f"trial {t.index} · loss {t.loss:.3f}<br>%{{y:.2f}} Nm<extra></extra>")
    fig.update_yaxes(title_text="torque (Nm)")
    return styled(p, phase_axis(fig), title, 320)


def power_figure(trial: trials.Trial, p: Palette = LIGHT) -> go.Figure:
    """Mean penalized power per gait-phase bin: where the loss is earned."""
    s, edges = trial.samples, np.linspace(0, 1, POWER_BINS + 1)
    fig = go.Figure().add_hline(y=0, line=dict(color=p.axis, width=1))
    for leg in (0, 1):
        bins = np.clip(np.digitize(cycles(s[f"phase_{leg}"]), edges) - 1, 0, POWER_BINS - 1)
        power = leg_power(s, leg, trial.settings.beta)
        mean = [power[bins == b].mean() if np.any(bins == b) else np.nan for b in range(POWER_BINS)]
        fig.add_scatter(x=(edges[1:] + edges[:-1]) / 2, y=mean, mode="lines+markers", name=f"leg {leg}",
                        line=dict(color=p.legs[leg], width=2), marker=dict(size=5),
                        hovertemplate="phase %{x:.2f}<br>%{y:.2f} W<extra></extra>")
    fig.update_yaxes(title_text=f"penalized power (W, negative ×{trial.settings.beta:g})")
    return styled(p, phase_axis(fig), "power over the gait cycle")


def portrait_figure(trial: trials.Trial, p: Palette = LIGHT) -> go.Figure:
    s, fig = trial.samples, go.Figure()
    for leg in (0, 1):
        fig.add_scatter(x=s[f"position_{leg}"], y=s[f"velocity_{leg}"], mode="lines", name=f"leg {leg}",
                        line=dict(color=p.legs[leg], width=1.5), hoverinfo="skip")
    fig.update_xaxes(title_text="position (rad)")
    fig.update_yaxes(title_text="velocity (rad/s)")
    return styled(p, fig, "phase portrait")


def trial_summary(trial: trials.Trial) -> str:
    s, p = trial.samples, trial.profile
    return (f"trial {trial.index} · loss {trial.loss:.3f} ± {trial.se:.3f} · {len(trials.gait_slices(s, trial.settings))} valid gaits, "
            f"{s['time'][-1] - s['time'][0]:.1f} s · {p.a.shape[-1]} atoms")


def css_variables(p: Palette) -> str:
    names = ("surface", "page", "ink", "ink_2", "muted", "axis", "border")
    return " ".join(f"--{n.replace('_', '-')}: {getattr(p, n)};" for n in names)


# the browser's light or dark setting
PAGE_CSS = f"""
:root {{ color-scheme: light; {css_variables(LIGHT)}
  --accent: #2a78d6; --ok: #1f8a4c; --warn: #b4630b; --err: #c93636; --busy: #2a78d6;
  --ok-soft: rgba(31,138,76,.09); --warn-soft: rgba(230,140,20,.11); --err-soft: rgba(201,54,54,.09);
  --busy-soft: rgba(42,120,214,.08); --shadow: 0 1px 2px rgba(0,0,0,.05), 0 2px 12px rgba(0,0,0,.04); }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; {css_variables(DARK)}
  --accent: #3987e5; --ok: #3fb771; --warn: #f0a63a; --err: #ef6464; --busy: #5a9cec;
  --ok-soft: rgba(63,183,113,.12); --warn-soft: rgba(240,166,58,.13); --err-soft: rgba(239,100,100,.13);
  --busy-soft: rgba(90,156,236,.12); --shadow: none; }} }}
""" + (WEB / "style.css").read_text()


def load_trials(exp_dir: str) -> list[trials.Trial]:
    """The folder's trials, none if it holds an older format."""
    try:
        return trials.load(exp_dir)
    except (KeyError, TypeError, ValueError, OSError):
        return []


def review_headline(history: list[trials.Trial]) -> str:
    """Trial count and the best trial."""
    best = min(history, key=lambda t: t.loss)
    return f"{len(history)} trials · best trial {best.index}, loss {best.loss:.3f}"


def animation_figure(history: list[trials.Trial], swing: float = 45.0, fps: float = 25.0) -> go.Figure:
    """Side-view animation of the gait: fixed hip pivot, two swinging legs, one segment per trial."""
    # resample every trial onto a uniform real-time grid
    segments = []
    for t in history:
        s = t.samples
        times = np.arange(s["time"][0], s["time"][-1], 1.0 / fps)
        keys = [f"{c}_{leg}" for c in ("position", "torque") for leg in (0, 1)]
        segments.append((t, {c: np.interp(times, s["time"], s[c]) for c in keys}))

    # one global angle scale so swings are comparable across trials
    peak = max(np.abs(np.stack([s["position_0"], s["position_1"]])).max() for _, s in segments)
    scale = math.radians(swing) / peak

    def frame_data(seg: dict, i: int) -> list[go.Scatter]:
        # angle 0 hangs straight down, positive swings forward (+x)
        angles = [scale * seg[f"position_{leg}"][i] for leg in (0, 1)]
        return [
            go.Scatter(x=[0, LEG_LENGTH * math.sin(a)], y=[0, -LEG_LENGTH * math.cos(a)], mode="lines+markers",
                       line=dict(color=LIGHT.legs[leg], width=6), marker=dict(size=[14, 8]),
                       name=f"leg {leg} (torque {seg[f'torque_{leg}'][i]:+.2f} Nm)")
            for leg, a in enumerate(angles)
        ]

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
    return fig.update_layout(
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
