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


################################################################################
# region Data


def cycles(phase: np.ndarray) -> np.ndarray:
    """Gait phase [rad] as a fraction of the cycle in [0, 1)."""
    return np.mod(phase, 2 * math.pi) / (2 * math.pi)


def columns(samples: list[dict]) -> dict[str, np.ndarray]:
    """Numeric sample fields as arrays, missing values as NaN."""
    if not samples:
        return {}
    keys = [k for k in samples[0] if k not in ("stage", "profile_id")]
    return {k: np.array([np.nan if s[k] is None else s[k] for s in samples], dtype=float) for k in keys}


def gait_period(s: dict[str, np.ndarray]) -> float | None:
    """Median time between phase wraps of leg 0."""
    ok = ~np.isnan(s["phase_0"])
    wraps = s["time"][ok][1:][np.diff(s["phase_0"][ok]) < -math.pi]
    return float(np.median(np.diff(wraps))) if len(wraps) >= 3 else None


def running_estimate(samples: dict[str, np.ndarray], points: int = 200) -> tuple[np.ndarray, ...]:
    """Running loss estimate and its SE, as the stopping rule sees it (trials.standard_error, vectorized).

    Evaluated at most at `points` sample counts (always including the last), so a refresh stays cheap.
    """
    power = trials.penalized_power(samples)
    counts = np.unique(np.linspace(2, len(power), points).astype(int))
    sums = np.concatenate([[0.0], np.cumsum(power)])
    squares = np.concatenate([[0.0], np.cumsum(power**2)])
    mean = sums[counts] / counts

    # few samples: plain standard error
    variance = np.maximum(squares[counts] - counts * mean**2, 0.0) / (counts - 1)
    se = np.sqrt(variance / counts)

    # otherwise batch means, split as np.array_split does: the first n % B batches one sample longer
    batches = trials.BATCHES
    q, r = np.divmod(counts[:, None], batches)
    j = np.arange(batches + 1)
    edges = j * q + np.minimum(j, r)
    batch_means = (sums[edges[:, 1:]] - sums[edges[:, :-1]]) / np.maximum(np.diff(edges, axis=1), 1)
    batched = batch_means.std(axis=1, ddof=1) / np.sqrt(batches)
    se = np.where(counts >= 2 * batches, batched, se)
    return counts - 1, -mean, se


def leg_power(samples: dict[str, np.ndarray], leg: int) -> np.ndarray:
    p = samples[f"torque_{leg}"] * samples[f"velocity_{leg}"]
    return np.where(p < 0, trials.BETA * p, p)


def cached(function):
    """Memoize on the history's (index, loss) pairs: trials themselves are not hashable."""
    memo = {}

    def wrapper(history: list[trials.Trial], *args):
        key = (tuple((t.index, t.loss) for t in history), args)
        if key not in memo:
            memo.clear()
            memo[key] = function(history, *args)
        return memo[key]

    return wrapper


@cached
def surrogate_check(history: list[trials.Trial], mode: str) -> dict | None:
    """The BO surrogate's held-out prediction of every trial."""
    if len(history) < 3:
        return None
    import jax.numpy as jnp
    from exopt.gaussian_process import GaussianProcess

    xs = EMBEDDINGS[mode](Profile.stack([t.profile for t in history]))
    gp = GaussianProcess.fit(xs, jnp.array([t.loss for t in history]), jnp.array([t.se for t in history]) ** 2)
    mean, sd = gp.leave_one_out()
    return dict(mean=np.asarray(mean), sd=np.asarray(sd), lengthscale=float(gp.lengthscale),
                signal_sd=math.sqrt(gp.signal), nugget_sd=math.sqrt(gp.nugget))


@cached
def profile_map(history: list[trials.Trial]) -> np.ndarray | None:
    """Principal coordinates of the profiles under the L2 distance the surrogate uses."""
    if len(history) < 3:
        return None
    from exopt.profiles import features

    z = np.asarray(features(Profile.stack([t.profile for t in history])))
    u, s, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
    return u[:, :2] * s[:2]


# endregion
################################################################################


################################################################################
# region Figures


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
        fig.add_scatter(x=PHASE_CYCLES, y=MAX_TORQUE * np.asarray(profile(PHASE)), mode="lines",
                        name="profile", line=dict(color=p.ink, width=2),
                        hovertemplate="phase %{x:.2f}<br>%{y:.2f} Nm<extra></extra>")
    fig.update_yaxes(title_text="torque (Nm)")
    return styled(p, phase_axis(fig), title)


def estimate_figure(title: str, samples: dict, rule: dict | None = None, p: Palette = LIGHT) -> go.Figure:
    """Running loss estimate of a segment with its ±2 SE band, against the stop threshold if any."""
    if len(samples.get("time", [])) < 2:
        return placeholder(p, "waiting for full-gain samples", title)
    at, loss, se = running_estimate(samples)
    t = samples["time"][at] - samples["time"][0]
    fig = go.Figure()
    fig.add_scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([loss + 2 * se, (loss - 2 * se)[::-1]]),
                    fill="toself", fillcolor=p.band, line=dict(width=0), name="±2 SE", hoverinfo="skip")
    fig.add_scatter(x=t, y=loss, mode="lines", name="estimate", line=dict(color=p.accent, width=2), customdata=se,
                    hovertemplate="%{x:.1f} s<br>loss %{y:.3f} ± %{customdata:.3f}<extra></extra>")
    if rule is not None:
        # the segment stops once its estimate crosses best + z * sqrt(se^2 + sd_best^2)
        best, sd, z = rule["best_mean"], rule["best_sd"], rule["z"]
        fig.add_scatter(x=t, y=np.full_like(t, best), mode="lines", name="best (posterior)",
                        line=dict(color=p.muted, width=1.5, dash="dash"), hoverinfo="skip")
        fig.add_scatter(x=t, y=best + z * np.hypot(se, sd), mode="lines", name=f"stop above ({z:g} sd)",
                        line=dict(color=p.legs[1], width=1.5, dash="dot"), hoverinfo="skip")
        note(p, fig, f"{(loss[-1] - best) / np.hypot(se[-1], sd):.1f} sd above the best (stops above {z:g})")
    fig.update_xaxes(title_text="time in segment (s)")
    fig.update_yaxes(title_text="loss")
    return styled(p, fig, title)


def live_figure(s: dict[str, np.ndarray], p: Palette = LIGHT) -> go.Figure | dict:
    """Position, velocity, torque and estimated phase of both legs.

    Built as a plain dict: plotly's validation of a four-row figure costs more than the 10 Hz refresh allows.
    """
    if not s:
        return placeholder(p, "no samples from the controller", height=420)
    rows = [("position", "rad"), ("velocity", "rad/s"), ("torque", "Nm"), ("phase", "cycles")]
    t, gap = s["time"][::LIVE_STRIDE].tolist(), 0.04
    height = (1 - gap * (len(rows) - 1)) / len(rows)
    data, layout = [], layout_style(p, height=420)
    for row, (kind, unit) in enumerate(rows, 1):
        suffix = "" if row == 1 else str(row)
        for leg in (0, 1):
            y = s[f"{kind}_{leg}"][::LIVE_STRIDE]
            data.append(dict(
                type="scatter", x=t, y=(cycles(y) if kind == "phase" else y).tolist(),
                xaxis=f"x{suffix}", yaxis=f"y{suffix}", mode="markers" if kind == "phase" else "lines",
                name=f"leg {leg}", legendgroup=f"leg {leg}", showlegend=row == 1,
                line=dict(color=p.legs[leg], width=1.5), marker=dict(color=p.legs[leg], size=2),
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
    fig.add_scatter(x=index, y=loss, mode="markers", name="trial", customdata=data, hovertemplate=hover,
                    error_y=dict(array=2 * se, color=p.axis, thickness=1.5, width=0),
                    marker=dict(size=9, color=p.muted, line=dict(color=p.surface, width=2)))
    fig.add_scatter(x=index[[best]], y=loss[[best]], mode="markers", name=f"best (trial {index[best]})",
                    customdata=data[[best]], hovertemplate=hover,
                    marker=dict(size=12, color=p.accent, line=dict(color=p.surface, width=2)))
    if selected in index:
        ring(p, fig, selected, loss[index == selected][0])
    fig.update_xaxes(title_text="trial", dtick=max(1, len(index) // 12))
    fig.update_yaxes(title_text="loss (±2 SE, batch means)")
    return styled(p, fig, title, 320)


def surrogate_figure(
    history: list[trials.Trial], selected: int | None, p: Palette = LIGHT, mode: str = "functional"
) -> go.Figure:
    title = "surrogate check: held-out prediction"
    check = surrogate_check(history, mode)
    if check is None:
        return placeholder(p, "needs 3 trials", title, 320)
    index = np.array([t.index for t in history])
    loss, mean, sd = np.array([t.loss for t in history]), check["mean"], check["sd"]
    se = np.array([t.se for t in history])
    span = [min(loss.min(), mean.min()), max(loss.max(), mean.max())]

    fig = go.Figure()
    fig.add_scatter(x=span, y=span, mode="lines", hoverinfo="skip", line=dict(color=p.axis, width=1.5, dash="dash"))
    fig.add_scatter(x=loss, y=mean, mode="markers", customdata=np.stack([index, sd, se], 1),
                    error_x=dict(array=2 * se, color=p.axis, thickness=1.5, width=0),
                    error_y=dict(array=2 * sd, color=p.axis, thickness=1.5, width=0),
                    marker=dict(size=9, color=p.accent, line=dict(color=p.surface, width=2)),
                    hovertemplate="trial %{customdata[0]}<br>observed %{x:.3f} ± %{customdata[2]:.3f}"
                                  "<br>predicted %{y:.3f} ± %{customdata[1]:.3f}<extra></extra>")
    if selected in index:
        ring(p, fig, loss[index == selected][0], mean[index == selected][0])
    note(p, fig, f"GP lengthscale {check['lengthscale']:.3g} · signal sd {check['signal_sd']:.3g}"
              f" · nugget sd {check['nugget_sd']:.3g}")
    fig.update_xaxes(title_text="observed loss (±2 SE)")
    fig.update_yaxes(title_text="predicted without it (±2 sd)")
    return styled(p, fig, title, 320, showlegend=False)


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
        fig.add_scatter(x=PHASE_CYCLES, y=MAX_TORQUE * np.asarray(t.profile(PHASE)), mode="lines", name=name,
                        showlegend=legend, line=dict(color=color, width=width), customdata=np.full(len(PHASE), t.index),
                        hovertemplate=f"trial {t.index} · loss {t.loss:.3f}<br>%{{y:.2f}} Nm<extra></extra>")
    fig.update_yaxes(title_text="torque (Nm)")
    return styled(p, phase_axis(fig), title, 320)


def map_figure(history: list[trials.Trial], selected: int | None, p: Palette = LIGHT) -> go.Figure:
    title = "profile space (L2 principal coordinates)"
    coords = profile_map(history)
    if coords is None:
        return placeholder(p, "needs 3 trials", title, 320)
    index = np.array([t.index for t in history])
    loss = np.array([t.loss for t in history])

    fig = go.Figure()
    fig.add_scatter(x=coords[:, 0], y=coords[:, 1], mode="lines", hoverinfo="skip", line=dict(color=p.grid, width=1))
    fig.add_scatter(
        x=coords[:, 0], y=coords[:, 1], mode="markers", customdata=np.stack([index, loss], 1),
        hovertemplate="trial %{customdata[0]}<br>loss %{customdata[1]:.3f}<extra></extra>",
        marker=dict(size=11, color=loss, colorscale=[[k / (len(p.ramp) - 1), c] for k, c in enumerate(p.ramp)],
                    line=dict(color=p.surface, width=2), colorbar=dict(title="loss", thickness=10, outlinewidth=0)),
    )
    if selected in index:
        ring(p, fig, *coords[index == selected][0])
    fig.update_xaxes(title_text="coordinate 1", zeroline=False)
    fig.update_yaxes(title_text="coordinate 2", zeroline=False, scaleanchor="x")
    return styled(p, fig, title, 320, showlegend=False)


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
    return (f"trial {trial.index} · loss {trial.loss:.3f} ± {trial.se:.3f} · {len(s['time'])} samples over "
            f"{s['time'][-1] - s['time'][0]:.1f} s · {p.atoms} atoms, ρ = {float(p.rho):.2f} rad")


# endregion
################################################################################


################################################################################
# region App

def css_variables(p: Palette) -> str:
    names = ("surface", "page", "ink", "ink_2", "muted", "axis", "border")
    return " ".join(f"--{n.replace('_', '-')}: {getattr(p, n)};" for n in names)


# light by default, dark when the OS asks for it, either when the toggle pins it
CSS = f"""
:root {{ color-scheme: light; {css_variables(LIGHT)} }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ color-scheme: dark; {css_variables(DARK)} }}
}}
:root[data-theme="dark"] {{ color-scheme: dark; {css_variables(DARK)} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font-family: {FONT}; }}
.page {{ max-width: 1440px; margin: 0 auto; padding: 16px; }}
header {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
header h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
header span {{ color: var(--ink-2); font-size: 13px; flex: 1; }}
h2 {{ font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-2); margin: 24px 0 8px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; }}
.tile, .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; min-width: 0; }}
.tile {{ padding: 10px 12px; }}
.card {{ padding: 4px; }}
.tile .label {{ font-size: 12px; color: var(--ink-2); }}
.tile .value {{ font-size: 20px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.tile .sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}
.tile.alert .value {{ color: #d03b3b; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 8px; margin-bottom: 8px; }}
.controls {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
button {{ font: inherit; font-size: 13px; padding: 4px 12px; border-radius: 6px; border: 1px solid var(--axis);
          background: var(--surface); color: var(--ink); cursor: pointer; }}
.controls span {{ color: var(--ink-2); font-size: 13px; }}
.dash-dropdown, .dash-dropdown-content, .dash-dropdown-search {{ background: var(--surface) !important;
  color: var(--ink) !important; border-color: var(--axis) !important; }}
.dash-dropdown-option:hover, .dash-dropdown-option.selected {{ background: var(--border) !important; }}
"""

# the theme the page shows: the toggle flips a pinned data-theme, else the OS setting decides
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
            profile_figure("profile and its atoms", trial.profile, trial.samples, "measured_torque", atoms=True, p=p),
            power_figure(trial, p=p),
            estimate_figure("convergence of the estimate", trial.samples, p=p),
            portrait_figure(trial, p=p),
            trial_summary(trial),
        )

    webbrowser.open(f"http://127.0.0.1:{port}")
    app.run(debug=False, port=port)


# endregion
################################################################################


################################################################################
# region Animation


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


# endregion
################################################################################


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

    if not args.animation:
        run_dashboard(args.exp_dir, args.host, args.port, args.warmup, args.mode)
        return
    out = f"{args.exp_dir}/animation.html"
    animation_figure(trials.load(args.exp_dir), args.swing, args.fps).write_html(out, auto_play=False)
    print(f"Wrote {out}")
    webbrowser.open(f"file://{os.path.abspath(out)}")


if __name__ == "__main__":
    main()
