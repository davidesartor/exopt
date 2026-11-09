"""start: connect to the exo, run an experiment, review any experiment; exo-review: its static page."""

from typing import NamedTuple, cast
from collections import deque
from html import escape
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import argparse
import json
import math
import os
import posixpath
import shlex
import signal
import subprocess
import sys
import threading
import time
import webbrowser

from exopt import MAX_TORQUE, link, trials
from exopt.tunnel import SSH_CONFIG, Tunnel, known_hosts, save_controller, save_host, saved_controllers
from exopt.profiles import EMBEDDINGS, Profile, atom

LIVE_WINDOW = 10.0  # [s] of leg traces in the running view
BUFFER_SECONDS = 60.0  # [s] of samples kept at full rate
TORQUE_WINDOW = 4.0  # [s] of live torque overlaid on the applied profile
LIVE_STRIDE = 3  # plot every 3rd sample of the traces, so a refresh stays cheap
NEW = "::new"  # the controller picker's configure new entry
REFRESH_MS = 1000  # live band refresh period (1 Hz)
FEED_MS = 1000 / 60  # live traces and segment estimate pulled and drawn at 60 Hz
LIVE_HEIGHT = 680  # [px] of the four rows of leg traces
FEED_POINTS = int((LIVE_WINDOW + 1.0) / 0.01 / LIVE_STRIDE)  # kept per trace by the page
LIVE_ROWS = (("position", "rad"), ("velocity", "rad/s"), ("torque", "Nm"), ("phase", "cycles"))
TRAJECTORY_KEYS = ("time", *(f"{kind}_{leg}" for kind, _ in LIVE_ROWS for leg in (0, 1)))
TRAJECTORY_STRIDE = 5  # every 5th sample (20 Hz) kept for the whole experiment view
WHOLE_POINTS = 4000  # at most per trace in the whole experiment view
WHOLE_REFRESH = 1.0  # [s] between redraws of the whole experiment view
STALE_AFTER = 2.0  # [s] without samples before the stream counts as lost
POWER_BINS = 32
PHASE = np.linspace(0.0, 2 * math.pi, 241)
PHASE_CYCLES = PHASE / (2 * math.pi)  # unwrapped, so curves end at 1 instead of jumping back to 0
LEG_LENGTH = 1.0  # animation units


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


columns = trials.columns


def gait_period(s: dict[str, np.ndarray]) -> float | None:
    """Median time between phase wraps of leg 0."""
    ok = ~np.isnan(s["phase_0"])
    wraps = s["time"][ok][1:][np.diff(s["phase_0"][ok]) < -math.pi]
    return float(np.median(np.diff(wraps))) if len(wraps) >= 3 else None


def running_estimate(samples: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample index, loss and SE at the end of each valid gait from the second on, as trials.gait_estimate gives them."""
    power, gaits = trials.penalized_power(samples), trials.gait_slices(samples)
    per_gait = np.array([-power[gait].mean() for gait in gaits])
    at, loss, se = [], [], []
    for k in range(2, len(gaits) + 1):
        at.append(gaits[k - 1].stop - 1)
        loss.append(-np.concatenate([power[gait] for gait in gaits[:k]]).mean())
        se.append(per_gait[:k].std(ddof=1) / np.sqrt(k))
    return np.array(at, int), np.array(loss), np.array(se)


def leg_power(samples: dict[str, np.ndarray], leg: int) -> np.ndarray:
    p = samples[f"torque_{leg}"] * samples[f"velocity_{leg}"]
    return np.where(p < 0, trials.BETA * p, p)


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
    profile: Profile | None,
    points: dict | None = None,
    label: str = "",
    atoms: bool = False,
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
        for k, (c, w) in enumerate(zip(np.asarray(profile.centers), np.asarray(profile.weights))):
            atom_torque = MAX_TORQUE * w * np.asarray(atom(profile.rho, PHASE - c))
            fig.add_scatter(x=PHASE_CYCLES, y=atom_torque, mode="lines", name="atoms", legendgroup="atoms",
                            showlegend=k == 0, line=dict(color=p.muted, width=1.5, dash="dash"),
                            hovertemplate=f"atom {k + 1}: center {cycles(c):.2f}, weight {w:+.3f}<extra></extra>")
    if profile is not None:
        fig.add_scatter(x=PHASE_CYCLES, y=MAX_TORQUE * np.clip(np.asarray(profile(PHASE)), -1.0, 1.0), mode="lines",
                        name="profile", line=dict(color=p.ink, width=2),
                        hovertemplate="phase %{x:.2f}<br>%{y:.2f} Nm<extra></extra>")
    fig.update_yaxes(title_text="torque (Nm)")
    return styled(p, phase_axis(fig), title)


def estimate_figure(title: str, samples: dict, rule: dict | None = None, p: Palette = LIGHT) -> go.Figure | dict:
    """Loss estimate of a segment ± STOP_Z SE at each valid gait, against the best observed profile's pessimistic mean.

    Built as a plain dict, cheap enough to redraw live."""
    at, loss, se = running_estimate(samples) if len(samples.get("time", [])) > 1 else ([], [], [])
    if len(at) == 0:
        return placeholder(p, "waiting for 2 valid gaits", title)
    z = trials.STOP_Z
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
        best = np.full_like(loss, trials.pessimistic_best(rule))
        data.append(line(best, f"best (posterior mean + {z:g} sd)", line=dict(color=p.muted, width=1.5, dash="dash"),
                         hoverinfo="skip"))
    data += [band(loss - z * se, loss + z * se, p.band, f"segment ± {z:g} SE (95%)"),
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
    # as applied: the controller clips the summed profile at MAX_TORQUE
    for t, color, width, name, legend in styles:
        applied = MAX_TORQUE * np.clip(np.asarray(t.profile(PHASE)), -1.0, 1.0)
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
        power = leg_power(s, leg)
        mean = [power[bins == b].mean() if np.any(bins == b) else np.nan for b in range(POWER_BINS)]
        fig.add_scatter(x=(edges[1:] + edges[:-1]) / 2, y=mean, mode="lines+markers", name=f"leg {leg}",
                        line=dict(color=p.legs[leg], width=2), marker=dict(size=5),
                        hovertemplate="phase %{x:.2f}<br>%{y:.2f} W<extra></extra>")
    fig.update_yaxes(title_text=f"penalized power (W, negative ×{trials.BETA:g})")
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
    return (f"trial {trial.index} · loss {trial.loss:.3f} ± {trial.se:.3f} · {len(trials.gait_slices(s))} valid gaits, "
            f"{s['time'][-1] - s['time'][0]:.1f} s · {p.atoms} atoms, ρ = {float(p.rho):.2f} rad")


def css_variables(p: Palette) -> str:
    names = ("surface", "page", "ink", "ink_2", "muted", "axis", "border")
    return " ".join(f"--{n.replace('_', '-')}: {getattr(p, n)};" for n in names)


# the browser's light or dark setting
CSS = f"""
:root {{ color-scheme: light; {css_variables(LIGHT)}
  --accent: #2a78d6; --ok: #1f8a4c; --warn: #b4630b; --err: #c93636; --busy: #2a78d6;
  --ok-soft: rgba(31,138,76,.09); --warn-soft: rgba(230,140,20,.11); --err-soft: rgba(201,54,54,.09);
  --busy-soft: rgba(42,120,214,.08); --shadow: 0 1px 2px rgba(0,0,0,.05), 0 2px 12px rgba(0,0,0,.04); }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; {css_variables(DARK)}
  --accent: #3987e5; --ok: #3fb771; --warn: #f0a63a; --err: #ef6464; --busy: #5a9cec;
  --ok-soft: rgba(63,183,113,.12); --warn-soft: rgba(240,166,58,.13); --err-soft: rgba(239,100,100,.13);
  --busy-soft: rgba(90,156,236,.12); --shadow: none; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font-family: {FONT}; font-size: 14px; }}
.page {{ max-width: 1360px; margin: 0 auto; padding: 20px 24px 48px; }}
header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }}
header h1 {{ font-size: 18px; margin: 0; font-weight: 650; }}
header > span {{ color: var(--ink-2); font-size: 13px; }}
.brand {{ font-weight: 700; font-size: 19px; letter-spacing: -.01em; color: var(--ink) !important; margin-right: 6px; }}
.spacer {{ flex: 1; }}
.pill {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border: 1px solid var(--border);
  border-radius: 999px; background: var(--surface); font-size: 13px; color: var(--ink-2); }}
.pill b {{ color: var(--ink); font-weight: 600; }}
.pill:empty {{ display: none; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }}
.dot.ok {{ background: var(--ok); }} .dot.busy {{ background: var(--busy); }} .dot.error {{ background: var(--err); }}
h2 {{ font-size: 13px; font-weight: 650; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-2); margin: 28px 0 10px; }}
.section-head {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 32px 0 10px; }}
.section-head h2 {{ margin: 0; }}
.hint {{ color: var(--muted); font-size: 13px; }}
.error-text {{ color: var(--err); font-size: 13px; }}

/* the status: steps, then the headline with its one action */
.tone-busy {{ --tone: var(--busy); --tone-soft: var(--busy-soft); }}
.tone-action {{ --tone: var(--warn); --tone-soft: var(--warn-soft); }}
.tone-ready {{ --tone: var(--accent); --tone-soft: var(--busy-soft); }}
.tone-ok {{ --tone: var(--ok); --tone-soft: var(--ok-soft); }}
.tone-error {{ --tone: var(--err); --tone-soft: var(--err-soft); }}
.stepper {{ display: flex; gap: 8px; list-style: none; margin: 0 0 12px; padding: 0; }}
.step {{ flex: 1; display: flex; align-items: center; gap: 8px; padding-top: 10px; border-top: 3px solid var(--border);
  color: var(--muted); font-size: 13px; }}
.step .num {{ width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: 12px;
  font-weight: 650; border: 1.5px solid currentColor; }}
.step.done {{ color: var(--ink-2); border-top-color: var(--ok); }}
.step.done .num {{ background: var(--ok); border-color: var(--ok); color: #fff; }}
.step.active {{ color: var(--ink); font-weight: 650; border-top-color: var(--tone); }}
.step.active .num {{ background: var(--tone); border-color: var(--tone); color: #fff; }}
.hero {{ display: flex; align-items: center; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  padding: 22px 26px; border-radius: 14px; background: var(--tone-soft); border: 1px solid var(--tone);
  border-left-width: 6px; margin-bottom: 16px; }}
.hero-text {{ flex: 1; min-width: 280px; }}
.eyebrow {{ font-size: 12px; font-weight: 650; letter-spacing: .07em; text-transform: uppercase; color: var(--tone); }}
.title {{ font-size: 28px; font-weight: 650; letter-spacing: -.01em; margin: 4px 0 6px; }}
.detail {{ color: var(--ink-2); font-size: 15px; max-width: 75ch; }}
.progress {{ margin-top: 14px; max-width: 560px; }}
.bar {{ position: relative; height: 10px; border-radius: 999px; background: var(--border); }}
.bar .fill {{ height: 100%; border-radius: 999px; background: var(--tone); transition: width .4s; }}
.bar .min {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--ink-2); }}
.caption {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--ink-2); margin-top: 6px; }}
.actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}

.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin-bottom: 10px; }}
.tile, .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; min-width: 0;
  box-shadow: var(--shadow); }}
.tile {{ padding: 12px 16px; }}
.card {{ padding: 6px; }}
.tile .label {{ font-size: 12px; color: var(--ink-2); text-transform: uppercase; letter-spacing: .05em; }}
.tile .value {{ font-size: 24px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.tile .sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 10px; margin-bottom: 10px; }}
.controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
.controls span {{ color: var(--ink-2); font-size: 13px; }}
.controls span.path {{ font: 12px ui-monospace, Menlo, monospace; }}

button, input {{ font: inherit; font-size: 13px; padding: 6px 14px; border-radius: 8px; border: 1px solid var(--axis);
  background: var(--surface); color: var(--ink); }}
button {{ cursor: pointer; font-weight: 550; }}
button:hover:not(:disabled) {{ border-color: var(--ink-2); }}
button:disabled {{ opacity: .4; cursor: default; }}
button.primary {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
button.primary:hover:not(:disabled) {{ filter: brightness(1.08); border-color: var(--accent); }}
button.danger {{ color: var(--err); border-color: var(--err); background: var(--surface); }}
.actions button {{ font-size: 15px; padding: 10px 22px; border-radius: 10px; }}

.segmented {{ display: inline-flex; gap: 4px; }}
.segmented label, .segmented .dash-options-list-option {{ padding: 3px 12px; border: 1px solid var(--border);
  border-radius: 999px; cursor: pointer; font-size: 13px !important; }}
.segmented label:has(input:checked), .segmented .dash-options-list-option:has(input:checked) {{
  background: var(--ink); color: var(--surface) !important; border-color: var(--ink); }}
.segmented input {{ display: none; }}

.console {{ margin-top: 36px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 10px 16px; }}
.console summary {{ cursor: pointer; color: var(--ink-2); font-size: 13px; }}
.log {{ font: 12px ui-monospace, Menlo, monospace; color: var(--ink-2); white-space: pre-wrap; margin: 10px 0;
  max-height: 16em; overflow-y: auto; }}

.landing {{ display: grid; place-items: center; min-height: 60vh; }}
.landing .card {{ width: min(100%, 520px); padding: 28px 32px; }}
.landing h3 {{ font-size: 20px; margin: 0 0 6px; }}
.landing p {{ color: var(--ink-2); margin: 0 0 16px; }}
.landing .controls button {{ font-size: 15px; padding: 9px 22px; }}
.modal {{ position: fixed; inset: 0; background: rgba(0, 0, 0, .45); z-index: 10; align-items: center;
  justify-content: center; padding: 16px; }}
.modal .card {{ width: min(100%, 640px); padding: 22px 26px; }}
.modal h3 {{ font-size: 18px; margin: 0 0 4px; }}
.modal p {{ color: var(--ink-2); font-size: 13px; margin: 0 0 14px; }}
.form {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px 18px; }}
.field label {{ display: block; font-size: 12px; font-weight: 600; color: var(--ink-2); margin-bottom: 5px; }}
.field .help {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.browser {{ max-height: 15em; overflow-y: auto; border: 1px solid var(--axis); border-radius: 8px; margin-bottom: 8px; }}
.browser button {{ display: block; width: 100%; text-align: left; border: 0; border-radius: 0; background: none;
  font: 12px ui-monospace, Menlo, monospace; padding: 4px 10px; }}
.browser button:hover {{ background: var(--page); }}
.browser button.picked {{ color: var(--accent); font-weight: 600; }}
.dash-input, .dash-input-element {{ background: var(--surface) !important; color: var(--ink) !important; font-size: 13px !important; }}
.dash-options-list-option {{ color: var(--ink) !important; font-size: 13px; }}
.dash-dropdown, .dash-dropdown-content, .dash-dropdown-search {{ background: var(--surface) !important;
  color: var(--ink) !important; border-color: var(--axis) !important; }}
.dash-dropdown-option:hover, .dash-dropdown-option.selected {{ background: var(--border) !important; }}
"""

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
FEED_JS = """
function(theme, span) {
    const S = window.exoFeed = window.exoFeed || {skip: 0, traces: 0, paused: false, gen: 0};
    Object.assign(S, {theme, span, cursor: null, shown: {}, gen: S.gen + 1});
    if (S.running) return null;
    S.running = true;
    const config = {displaylogo: false, responsive: true};
    const hold = on => {
        if (S.paused === on) return;
        S.paused = on;
        dash_clientside.set_props("live-hint", {children: on ? "held while zoomed · double-click to follow" : ""});
    };
    const draw = (gd, feed) => {
        if (feed.figure) {
            Plotly.react(gd, feed.figure.data, feed.figure.layout,
                         {...config, doubleClick: "autosize", modeBarButtonsToRemove: ["resetScale2d"]});
            Object.assign(S, {traces: feed.figure.data.length, skip: 0});
            hold(false);
            if (!gd.exoZoom) {
                gd.exoZoom = true;
                gd.on("plotly_relayout", e => {
                    const keys = Object.keys(e || {});
                    if (keys.some(k => k.endsWith("autorange") && e[k])) {
                        hold(false);
                        Object.assign(S, {cursor: null, gen: S.gen + 1});  // the whole figure again
                    } else if (keys.some(k => k.includes("range["))) hold(true);
                });
            }
            return feed.latest;
        }
        // a remounted graph lost its traces: ask for the whole figure again
        if (!gd.data || gd.data.length !== S.traces) return null;
        const keep = feed.t.map((_, i) => (S.skip + i) % STRIDE === 0);
        S.skip = (S.skip + feed.t.length) % STRIDE;
        const pick = a => a.filter((_, i) => keep[i]);
        const t = pick(feed.t), xs = [], ys = [];
        // phase lines break where the phase wraps
        feed.y.forEach((y, i) => {
            const x = [], v = [];
            let last = gd.data[i].y[gd.data[i].y.length - 1];
            pick(y).forEach((value, j) => {
                if (PHASE.includes(i) && value !== null && last !== null && value < last - 0.5) {
                    x.push(t[j]); v.push(null);
                }
                x.push(t[j]); v.push(value); last = value;
            });
            xs.push(x); ys.push(v);
        });
        // the new range rides along with the extension's redraw: one redraw per chunk
        Object.assign(gd.layout.xaxis, {range: [feed.latest - WINDOW, feed.latest], autorange: false});
        if (t.length) Plotly.extendTraces(gd, {x: xs, y: ys}, feed.y.map((_, i) => i), POINTS);
        return feed.latest;
    };
    // a request continues from the reply before it, so the next one is sent while this one is drawn
    const request = () => {
        const gen = S.gen;
        const query = new URLSearchParams({theme: S.theme || "", span: S.span || "window",
            paused: S.paused ? 1 : 0, cursor: S.cursor ?? "", ...S.shown});
        return fetch("live-feed?" + query).then(r => r.json()).then(feed => ({feed, gen}), () => ({gen: -1}));
    };
    const step = async () => {
        const started = performance.now();
        let wait = PERIOD;
        try {
            const gd = document.getElementById("live");
            if (!gd || !window.Plotly || gd.offsetParent === null) {
                wait = 500;  // not on the session page: redraw whole figures once it shows again
                Object.assign(S, {cursor: null, shown: {}, gen: S.gen + 1, next: null});
            } else {
                const {feed, gen} = await (S.next || request());
                S.next = null;
                if (!feed) {
                    wait = 500;
                } else if (gen === S.gen) {  // else the view or theme changed meanwhile
                    if (feed.traces) S.cursor = feed.traces.latest ?? null;
                    for (const [id, panel] of Object.entries(feed.panels || {})) S.shown[id] = panel.key;
                    if (!feed.idle) S.next = request();
                    if (feed.traces && draw(gd, feed.traces) === null) {
                        Object.assign(S, {cursor: null, gen: S.gen + 1});  // lost its traces: the whole figure
                    }
                    for (const [id, panel] of Object.entries(feed.panels || {})) {
                        const div = document.getElementById(id);
                        if (div) Plotly.react(div, panel.figure.data, panel.figure.layout, config);
                    }
                }
                if (feed && feed.idle) wait = 500;
            }
        } catch (error) {
            wait = 500;
        }
        setTimeout(step, Math.max(0, wait - (performance.now() - started)));
    };
    step();
    return null;
}
""".replace("STRIDE", str(LIVE_STRIDE)).replace("POINTS", str(FEED_POINTS)).replace("WINDOW", str(LIVE_WINDOW)) \
    .replace("PERIOD", str(FEED_MS)) \
    .replace("PHASE", str([2 * row + leg for row, (kind, _) in enumerate(LIVE_ROWS) if kind == "phase" for leg in (0, 1)]))
CLICKABLE = ("loss", "profiles")
TRIAL = ("profile", "power", "convergence", "portrait")


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


def experiment_folders(experiments: str) -> list[str]:
    """Experiment folders, newest first."""
    root = Path(experiments)
    return sorted((str(d) for d in root.iterdir() if (d / "data").is_dir()), reverse=True) if root.is_dir() else []


STEPS = ("Connect", "Walk", "Experiment")
MOVING_SPEED = 0.15  # [rad/s] the controller's still-standing threshold
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


def trial_status(last: dict, current: dict | None, history: list[trials.Trial], segment: dict) -> Status:
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
    loss, se, gaits = trials.gait_estimate(segment)
    reason = trials.stop_reason(loss, se, gaits, rule) or ""
    detail = f"Stopping at the end of this gait: {STOP_WORDS[reason]}." if reason else "Keep walking."
    return Status(2, "ok", f"Measuring trial {index}", detail, "stop", eyebrow,
                  (gaits, rule.get("min_gaits", 0), rule.get("max_gaits", 1), reason))


def live_tiles(s: dict, last: dict | None, segment: dict, history: list[trials.Trial]) -> list[tuple]:
    """(label, value, detail) for each tile."""
    period = gait_period(s) if s else None
    gain = last["gain"] if last else None
    tiles = [
        ("gait period", f"{period:.2f} s", f"{60 / period:.0f} strides/min") if period else ("gait period", "—", "not walking"),
        ("torque", f"{100 * gain:.0f}%", "full" if gain >= 1 else "ramping" if gain > 0 else "off") if last
        else ("torque", "—", "no data"),
    ]
    _, _, gaits = trials.gait_estimate(segment) if len(segment.get("time", [])) > 1 else (0, 0, 0)
    if gaits >= 2:
        loss, se, _ = trials.gait_estimate(segment)
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


def controller_command(script: str) -> str:
    """Shell command running a controller script in its checkout's environment, with uv where its installer puts it."""
    folder, name = posixpath.split(script)
    return (f'cd {shlex.quote(folder)} && PATH="$HOME/.local/bin:$PATH" '
            f"exec uv run python {shlex.quote(name)}")


def run_dashboard(exp_dir: str, host: str, port: int, mode: str, exo: tuple[str, str, int], atoms: int):
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
    run = dict(exp_dir=exp_dir, mode=mode, process=None, started=None, t0=None)
    # the session shown (None: the landing page; name None: review only) and the configure window's state
    state: dict = dict(session=None, tunnel=None, script=None, autostart=False)
    wizard: dict = dict(step=0, tunnel=None, host=None, fresh=False)

    tile = lambda label, value, sub: html.Div(className="tile", children=[
        html.Div(label, className="label"), html.Div(value, className="value"), html.Div(sub, className="sub"),
    ])
    graph = lambda id: html.Div(className="card", children=dcc.Graph(id=id, config=dict(displaylogo=False)))
    grid = lambda *ids: html.Div(className="grid", children=[graph(id) for id in ids])
    app = Dash(__name__, title="exopt", update_title=None)
    app.index_string = app.index_string.replace("{%css%}", "{%css%}<style>" + CSS + "</style>")
    labeled = lambda label, field: html.Span([label, " ", field])
    field = lambda label, control, help="": html.Div(className="field", children=[
        html.Label(label), control, html.Div(help, className="help") if help else None])
    number = lambda id, value, width, **kw: dcc.Input(id=id, type="number", value=value, style=dict(width=width), **kw)
    exo_host, exo_user, exo_port = exo
    dropdown = lambda id, width, **kw: dcc.Dropdown(id=id, clearable=False, style=dict(
        width=width, display="inline-block", verticalAlign="middle"), **kw)
    hide = dict(display="none")
    landing = html.Div(id="landing", className="landing", children=html.Div(className="card", children=[
        html.H3("Connect to the exoskeleton"),
        html.P("Pick a controller and connect. To add one, choose “+ configure new” in the list."),
        dropdown("controller-pick", "100%", placeholder="configure a controller"),
        html.Div(className="controls", style=dict(marginTop="16px"), children=[
            html.Button("Connect", id="connect", className="primary"), html.Span(id="landing-note"),
        ]),
    ]))
    configure_new = html.Div(id="wizard", className="modal", style=hide, children=[
        html.Div(className="card", children=[
            html.H3("Configure a controller"),
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
            html.Div(id="wizard-script", style=hide, children=[
                html.P("2 / 3 · controller script, in the host's copy of exopt: src/exopt/controller.py drives the "
                       "legs, mock_controller.py simulates them."),
                html.Div(className="controls", children=[
                    html.Button("↑", id="browse-up", title="parent folder"),
                    html.Span(id="browse-path", className="path"),
                ]),
                html.Div(id="browse-list", className="browser"),
                html.Div(className="controls", children=html.Span(id="script-label")),
            ]),
            html.Div(id="wizard-name", style=hide, children=[
                html.P("3 / 3 · nickname, under which this host and script are saved."),
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
                field("Surrogate", dcc.Dropdown(id="exp-mode", options=list(EMBEDDINGS), value=mode, clearable=False),
                      "functional is the default; coefficients is the baseline"),
                field("Atoms per profile", number("exp-atoms", atoms, "100%", min=1, step=1)),
                field("Initial design", number("exp-n", 4, "100%", min=1, step=1), "random profiles before optimizing"),
                field("Min valid gaits per profile", number("exp-min-gaits", 5, "100%", min=2, step=1)),
                field("Max valid gaits per profile", number("exp-max-gaits", 20, "100%", min=1, step=1)),
            ]),
            html.Div(className="controls", style=dict(marginTop="20px"), children=[
                html.Button("Start", id="start", className="primary"), html.Button("Cancel", id="cancel"),
                html.Span(id="exp-note", className="error-text"),
            ]),
        ]),
    ])
    status_section = html.Div(id="status", className="status tone-busy", children=[
        html.Ol(id="stepper", className="stepper"),
        html.Div(className="hero", children=[
            html.Div(className="hero-text", children=[
                html.Div(id="hero-eyebrow", className="eyebrow"), html.Div(id="hero-title", className="title"),
                html.Div(id="hero-detail", className="detail"), html.Div(id="hero-progress"),
            ]),
            html.Div(className="actions", children=[
                html.Button("Start experiment", id="configure", className="primary", style=hide),
                html.Button("Restart controller", id="controller-start", className="primary", style=hide),
                html.Button("Reconnect", id="reconnect", className="primary", style=hide),
                html.Button("Stop experiment", id="stop", className="danger", style=hide),
            ]),
        ]),
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
    console = html.Details(className="console", children=[
        html.Summary(id="console-summary"),
        html.Pre(id="controller-log", className="log"),
        html.Div(className="controls", children=[
            html.Button("Stop controller", id="controller-stop"),
            html.Span("Sends ctrl-c: the drives are released.", className="hint"),
        ]),
    ])
    app.layout = html.Div(className="page", children=[
        html.Header(className="topbar", children=[
            html.Span("exopt", className="brand"),
            html.Span(id="session-pill", className="pill", style=hide),
            html.Span(id="experiment-pill", className="pill", style=hide),
            html.Span(className="spacer"),
            html.Button("Disconnect", id="disconnect", style=hide),
        ]),
        landing,
        html.Div(id="session", style=hide, children=[status_section, live_section, review_section, console]),
        config, configure_new,
        dcc.Store(id="selected"), dcc.Store(id="latest-trial"),
        dcc.Store(id="theme"),
        dcc.Store(id="live-loop"),
        dcc.Store(id="browse-dir"), dcc.Store(id="script"), dcc.Store(id="ssh-checked"),
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
                    profile = Profile.from_dict(current) if current else None
                    profile_bases.clear()
                    profile_bases[current_id, theme] = as_dict(
                        profile_figure(titles["now-profile"], profile, empty, "torque", p=p))
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
        segment = [x for x in data if current and x["profile_id"] == current_id and x["gain"] >= 1][link.WARMUP:]
        key = f"{current_id}:{len(segment)}"
        if shown.get("segment") != key:
            figure = estimate_figure(titles["segment"], columns(segment), current and current.get("stop_rule"), p=p)
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
            options.append(dict(label=f"{c['name']} · {where} · {posixpath.basename(c['script'])}", value=c["name"]))
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
        Output("wizard-ssh", "style"), Output("wizard-script", "style"), Output("wizard-name", "style"),
        Output("wizard-back", "style"), Output("wizard-next", "children"), Output("wizard-note", "children"),
        Output("nickname", "value"), Output("new-password", "value"), Output("config-alias", "options"),
        Input("controller-pick", "value"), Input("wizard-next", "n_clicks"), Input("wizard-back", "n_clicks"),
        Input("wizard-cancel", "n_clicks"), Input("ssh-checked", "data"),
        State("add-source", "value"), State("config-alias", "value"), State("new-name", "value"),
        State("new-address", "value"), State("new-user", "value"), State("new-port", "value"),
        State("new-password", "value"), State("script", "data"), State("nickname", "value"),
    )
    def configure(picked, _next, _back, _cancel, _, source, config_alias, name, address, user, port, password, script,
                  nickname):
        """The configure window: 1 ssh host (connected to check it), 2 controller script, 3 nickname, then saved."""
        trigger, step, note = ctx.triggered_id, wizard["step"], ""
        value = nick = password_field = no_update
        connecting = cast(Tunnel | None, wizard["tunnel"])
        last_used = next((c["name"] for c in saved_controllers()), None)
        if trigger is None:
            value = last_used
        elif trigger == "ssh-checked":
            if step != 1 or connecting is None or not (connecting.connected or connecting.error):
                return (no_update,) * 12
            if connecting.connected:
                wizard.update(step=2, fresh=True)
            else:
                note = f"ssh to {connecting.target} failed: {connecting.error}"
                wizard["tunnel"] = None
        elif trigger == "controller-pick" and picked == NEW:
            close_wizard_tunnel()
            wizard.update(step=1, host=None)
            value, password_field = last_used, ""
        elif trigger == "wizard-cancel":
            close_wizard_tunnel()
            wizard["step"] = 0
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
            if script:
                wizard["step"], nick = 3, wizard["host"]
            else:
                note = "pick the controller script"
        elif trigger == "wizard-next" and step == 3:
            nickname = (nickname or "").strip()
            if not nickname or nickname.startswith("::"):
                note = "a nickname is needed"
            elif not script:
                note = "no script picked: go back to pick it"
            else:
                save_controller(nickname, wizard["host"], script)
                close_wizard_tunnel()
                wizard["step"], value = 0, nickname

        step = wizard["step"]
        shown = lambda s: dict() if step == s else hide
        others = [dict(label=f"{a} · {h['user']}@{h['hostname']}", value=a) for a, h in known_hosts().items()]
        return (picker_options(), value, dict(display="flex") if step else hide, shown(1), shown(2), shown(3),
                dict() if step > 1 else hide, "Save" if step == 3 else "Next", note, nick, password_field, others)

    @app.callback(Output("from-config", "style"), Output("from-manual", "style"), Input("add-source", "value"))
    def add_source(source):
        return (dict(), hide) if source == "config" else (hide, dict())

    @app.callback(
        Output("browse-dir", "data"), Output("browse-list", "children"), Output("browse-path", "children"),
        Output("script", "data"), Output("script-label", "children"),
        Input("wizard-script", "style"), Input("browse-up", "n_clicks"), Input(dict(entry=ALL, folder=ALL), "n_clicks"),
        State("browse-dir", "data"), State("script", "data"),
    )
    def browse(card, _up, _entries, folder, script):
        """The host's files over sftp: folders open, a .py file becomes the controller script."""
        current = cast(Tunnel | None, wizard["tunnel"])
        trigger = ctx.triggered_id
        if card.get("display") == "none" or current is None or not current.connected:
            return (no_update,) * 5
        if wizard["fresh"]:
            wizard["fresh"], folder = False, None
        # a listing exists already: only clicks change it (freshly listed entries fire with no clicks)
        elif trigger in (None, "wizard-script") or not ctx.triggered[0]["value"]:
            return (no_update,) * 5

        # start in the folder of a script configured on the same host, else the home folder
        if folder is None:
            known = [c["script"] for c in saved_controllers() if c["host"] == wizard["host"]]
            script, target = None, posixpath.dirname(known[0]) if known else "."
        elif trigger == "browse-up":
            target = posixpath.dirname(folder.rstrip("/")) or "/"
        elif trigger["folder"]:
            target = posixpath.join(folder, trigger["entry"])
        else:
            script, target = posixpath.join(folder, trigger["entry"]), folder
        try:
            folder, entries = current.browse(target)
        except OSError as error:
            try:
                if folder is not None or target == ".":
                    raise
                folder, entries = current.browse(".")  # that folder is gone
            except OSError:
                return no_update, no_update, f"{target}: {error}", no_update, no_update
        rows = [html.Button(name + ("/" if is_folder else ""), id=dict(entry=name, folder=is_folder),
                            className="picked" if script == posixpath.join(folder, name) else "")
                for name, is_folder in entries if is_folder or name.endswith(".py")]
        label = f"script: {script}" if script else "no script picked"
        return folder, rows or [html.Span("no folders or .py files", className="path")], folder, script, label

    def experiment_running() -> bool:
        return run["process"] is not None and run["process"].poll() is None

    def open_session(name: str) -> str:
        """Open a session (a configured controller's: its tunnel, then its script); why not, if it cannot."""
        config = next((c for c in saved_controllers() if c["name"] == name), None)
        found = known_hosts().get(config["host"]) if config else None
        if config is None or found is None:
            return f"{name}: its ssh host is no longer in {SSH_CONFIG}, configure it again"
        save_controller(**config)  # last used first
        buffer.clear()
        trajectory.clear()
        opened = Tunnel(found["hostname"], found["user"], found["port"], found["password"],
                        (link.SAMPLE_PORT, link.PROFILE_PORT), tuple(found["keys"]))
        state.update(session=dict(name=name), tunnel=opened, script=config["script"], autostart=True)
        return ""

    def close_session():
        if state["tunnel"] is not None:
            state["tunnel"].close()  # ctrl-c to the controller first
        state.update(session=None, tunnel=None, script=None, autostart=False)

    @app.callback(
        Output("landing", "style"), Output("session", "style"), Output("disconnect", "style"),
        Output("session-pill", "style"), Output("experiment-pill", "style"), Output("landing-note", "children"),
        Input("connect", "n_clicks"), Input("disconnect", "n_clicks"), Input("fast", "n_intervals"),
        State("controller-pick", "value"),
    )
    def session(_connect, _disconnect, _, picked):
        """The landing page, or the session (a reloaded page returns to the open one)."""
        note = no_update if ctx.triggered_id == "fast" else ""
        if ctx.triggered_id == "connect" and state["session"] is None:
            note = open_session(picked) if picked not in (None, NEW) else "configure a controller first"
        elif ctx.triggered_id == "disconnect" and not experiment_running():
            close_session()
        if state["session"] is None:
            return dict(), hide, hide, hide, hide, note
        return hide, dict(), dict(), dict(), dict(), note

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

    def session_status(last: dict | None, current: dict | None, history: list[trials.Trial], segment: dict,
                       window: dict) -> Status:
        """Where the session is, from the tunnel, the controller script, its stream and the experiment."""
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
            return trial_status(last, current, history, segment)
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
        Input("controller-start", "n_clicks"), Input("controller-stop", "n_clicks"),
        Input("start", "n_clicks"), Input("stop", "n_clicks"), Input("reconnect", "n_clicks"),
        Input("fast", "n_intervals"),
        State("exp-name", "value"), State("exp-mode", "value"), State("exp-atoms", "value"), State("exp-n", "value"),
        State("exp-min-gaits", "value"), State("exp-max-gaits", "value"),
    )
    def status(*args):
        """Act on a click, then render the session from server state (so a dropped response heals on the next tick)."""
        name, chosen_mode, atoms, n, min_gaits, max_gaits = args[-6:]
        if state["session"] is None:
            return (no_update,) * 17
        tunnel, trigger = cast(Tunnel, state["tunnel"]), ctx.triggered_id

        # clicks: reconnect, the controller script, the experiment
        if trigger == "reconnect" and not experiment_running():
            controller_name = state["session"]["name"]
            close_session()
            open_session(controller_name)
            tunnel = cast(Tunnel, state["tunnel"])
        command = tunnel.command
        running = command is not None and command.running()
        connected = tunnel.connected and tunnel.alive()
        if (trigger == "controller-start" or (state["autostart"] and command is None)) and connected:
            if running:
                command.interrupt()
            command, running, state["autostart"] = tunnel.start(controller_command(state["script"])), True, False
        elif trigger == "controller-stop" and running:
            command.interrupt()
        elif trigger == "start" and not experiment_running() and name and name.strip():
            min_gaits = max(min_gaits or 5, 2)
            options = dict(mode=chosen_mode, atoms=atoms or 1, n=n or 4, min_gaits=min_gaits,
                           max_gaits=max(max_gaits or 20, min_gaits))
            folder = os.path.join(experiments, f"{time.strftime('%Y-%m-%d')}-{name.strip().replace(' ', '-')}")
            run.update(process=launch_bo_loop(folder, host, options), started=time.monotonic(), exp_dir=folder,
                       mode=chosen_mode, t0=buffer[-1]["time"] if buffer else None)
        elif trigger == "stop" and experiment_running():
            # an interrupt, so bo-loop releases the profile and the torque ramps down
            run["process"].send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)

        # what is going on
        data = list(buffer)
        last = data[-1] if data and time.monotonic() - received["at"] < STALE_AFTER else None
        window = columns([x for x in data if x["time"] > data[-1]["time"] - LIVE_WINDOW]) if data else {}
        current = trials.load_current(run["exp_dir"]) if experiment_running() else None
        history = load_trials(run["exp_dir"])
        segment = columns([x for x in data if current and x["profile_id"] == current["id"] and x["gain"] >= 1]
                          [link.WARMUP:])
        now = session_status(last, current, history, segment, window)

        # the steps, the headline and its one action
        steps = []
        for k, step_name in enumerate(STEPS):
            kind = "done" if k < now.step else ("active" if k == now.step else "todo")
            steps.append(html.Li(className=f"step {kind}", children=[
                html.Span("✓" if kind == "done" else str(k + 1), className="num"), html.Span(step_name)]))
        shown = lambda id: dict() if id == now.action or (id == "stop" and experiment_running()) else hide

        # the controller's shell, folded away
        log = "\n".join(command.output.rstrip("\n").split("\n")[-12:]) if command is not None else ""
        if command is not None and not running:
            exit_status = command.exit_status()
            log += "\n· controller exited" + (f" ({exit_status})" if exit_status not in (None, -1) else "")
        summary = f"Controller output · {'running' if running else 'stopped'} · {posixpath.basename(state['script'])}"

        dot = {"error": "error", "busy": "busy"}.get(now.tone, "ok") if now.step or now.tone != "busy" else "busy"
        pill = [html.Span(className=f"dot {dot}"), html.B(state["session"]["name"]), tunnel.target]
        experiment = [html.Span("experiment"), html.B(os.path.basename(run["exp_dir"])), run["mode"]] \
            if run["process"] is not None else []
        return (
            f"status tone-{now.tone}", steps, now.eyebrow or f"step {now.step + 1} of {len(STEPS)}", now.title,
            now.detail, progress_bar(*now.progress) if now.progress else None,
            *(shown(id) for id in actions), log, summary, not running,
            [tile(*t) for t in live_tiles(window, last, segment, history)], pill, experiment, experiment_running(),
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
        return options, no_update if ctx.triggered_id == "review-dir" else folder, headline, dict() if history else hide

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
            profile_figure("profile and its atoms", trial.profile, trial.samples, "measured_torque", atoms=True, p=p),
            power_figure(trial, p=p),
            estimate_figure("convergence of the estimate", trial.samples, p=p),
            portrait_figure(trial, p=p),
            trial_summary(trial),
        )

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


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


# figures are built light; the page swaps each light palette color for its dark counterpart
DARK_COLORS = {
    light: dark
    for light_value, dark_value in zip(LIGHT, DARK)
    for light, dark in zip(*((v,) if isinstance(v, str) else v for v in (light_value, dark_value)))
    if light != dark
}

PAGE_JS = """
const data = JSON.parse(document.getElementById("review-data").textContent);
const darkColors = JSON.parse(document.getElementById("dark-colors").textContent);
const swap = new RegExp(Object.keys(darkColors).map((c) => c.replace(/[()|.]/g, "\\\\$&")).join("|"), "g");
const config = { displaylogo: false, responsive: true };

const theme = () => (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
const themed = (figure) => JSON.parse(theme() === "dark" ? figure.replace(swap, (c) => darkColors[c]) : figure);
const draw = (id, figure) => { const f = themed(figure); Plotly.react(id, f.data, f.layout, config); };

let shown = data.best;  // position of the trial shown
function showTrial(position) {
  shown = Math.min(Math.max(position, 0), data.trials.length - 1);
  const trial = data.trials[shown];
  document.getElementById("summary").textContent = trial.summary;
  for (const name of Object.keys(trial.figures)) draw("trial-" + name, trial.figures[name]);
}
function drawAll() {
  for (const name of Object.keys(data.optimization)) draw(name, data.optimization[name]);
  if (data.trials.length) showTrial(shown);
}

matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
drawAll();

// a click on any trial marker or line shows that trial
const positions = new Map(data.trials.map((trial, position) => [trial.index, position]));
for (const name of Object.keys(data.optimization)) {
  document.getElementById(name).on("plotly_click", (event) => {
    const custom = event.points[0].customdata;
    const index = Number(Array.isArray(custom) ? custom[0] : custom);
    if (positions.has(index)) showTrial(positions.get(index));
  });
}
"""


def review_html(exp_dir: str, mode: str) -> str:
    """The review page of the experiment in exp_dir, plotly included, so it opens offline."""
    history = trials.load(exp_dir)
    name = Path(exp_dir).resolve().name
    to_json = lambda figure: go.Figure(figure).to_json()

    optimization = (loss_figure, profiles_figure)
    trial_pages = [
        dict(index=t.index, summary=trial_summary(t), figures=dict(zip(TRIAL, map(to_json, (
            profile_figure("profile and its atoms", t.profile, t.samples, "measured_torque", atoms=True),
            power_figure(t),
            estimate_figure("convergence of the estimate", t.samples),
            portrait_figure(t),
        )))))
        for t in history
    ]
    best = min(range(len(history)), key=lambda k: history[k].loss, default=0)
    data = dict(
        optimization={key: to_json(figure(history, None)) for key, figure in zip(CLICKABLE, optimization)},
        trials=trial_pages, best=best,
    )
    # keep "</script>" inside the data from closing its tag
    embed = lambda value: json.dumps(value).replace("</", "<\\/")

    headline = f"{review_headline(history)} · {mode}" if history else f"no trials · {mode}"
    graphs = lambda ids, prefix="": "".join(f'<div class="card"><div id="{prefix}{i}"></div></div>' for i in ids)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(name)} review</title>
<style>{CSS}
select {{ font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--axis);
  background: var(--surface); color: var(--ink); }}</style>
<script>{get_plotlyjs()}</script>
</head><body><div class="page">
<header><h1>{escape(name)}</h1>
<span>{escape(headline)} · written {time.strftime("%Y-%m-%d %H:%M")}</span>
</header>
<h2>optimization</h2>
<div class="grid">{graphs(CLICKABLE)}</div>
<h2>trial</h2>
<div class="controls">
<span id="summary"></span></div>
<div class="grid">{graphs(TRIAL[:2], "trial-")}</div><div class="grid">{graphs(TRIAL[2:], "trial-")}</div>
</div>
<script type="application/json" id="review-data">{embed(data)}</script>
<script type="application/json" id="dark-colors">{embed(DARK_COLORS)}</script>
<script>{PAGE_JS}</script>
</body></html>
"""


def write_review(exp_dir: str, mode: str) -> Path:
    """Write <exp_dir>/dashboard.html."""
    out = Path(exp_dir) / "dashboard.html"
    out.write_text(review_html(exp_dir, mode))
    return out


def review_main():
    """exo-review: rewrite <exp_dir>/dashboard.html."""
    parser = argparse.ArgumentParser(description="exo-review: the static review page of an experiment.")
    parser.add_argument("exp_dir", help="experiment folder")
    parser.add_argument("--mode", choices=EMBEDDINGS, default="functional", help="surrogate inputs (as bo-loop)")
    args = parser.parse_args()
    print(f"Wrote {write_review(args.exp_dir, args.mode)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", nargs="?", default="experiments/mock", help="experiment folder")
    parser.add_argument("--host", default="localhost", help="controller host (tunnel endpoint)")
    parser.add_argument("--port", type=int, default=8050, help="dashboard port")
    parser.add_argument("--exo-host", default="", help="manual new host's default address")
    parser.add_argument("--exo-user", default="pi", help="ssh user on the Pi")
    parser.add_argument("--exo-port", type=int, default=22, help="ssh port on the Pi")
    parser.add_argument("--mode", choices=EMBEDDINGS, default="functional", help="surrogate inputs (as bo-loop)")
    parser.add_argument("--atoms", type=int, default=2, help="new experiment default (as bo-loop)")
    parser.add_argument("--animation", action="store_true", help="export the gait animation instead")
    parser.add_argument("--swing", type=float, default=45.0, help="animation max swing angle (deg)")
    parser.add_argument("--fps", type=float, default=25.0, help="animation frame rate")
    args = parser.parse_args()

    if not args.animation:
        run_dashboard(args.exp_dir, args.host, args.port, args.mode, (args.exo_host, args.exo_user, args.exo_port), args.atoms)
        return
    out = f"{args.exp_dir}/animation.html"
    animation_figure(trials.load(args.exp_dir), args.swing, args.fps).write_html(out, auto_play=False)
    print(f"Wrote {out}")
    webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
