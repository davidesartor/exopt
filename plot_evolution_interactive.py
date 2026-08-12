
import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from src import experiment_io

REACHABLE = (0.137, 0.908)


def load_completed(exp_dir):
    paired = sorted(
        experiment_io.config_numbers(exp_dir) & experiment_io.run_numbers(exp_dir)
    )
    indices, fs, ys = [], [], []
    for i in paired:
        try:
            y = experiment_io.read_result(exp_dir, i)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"  skipping run_{i}.txt: no table in it yet")
            continue
        indices.append(i)
        fs.append(experiment_io.read_config_function(exp_dir, i))
        ys.append(y)
    return np.array(indices), fs, np.array(ys)


def build_figure(indices, us, curves, ys, running_best, i_best_at, exp_name, band):
    n = len(ys)
    it = np.asarray(indices)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.42, 0.58],
        subplot_titles=["objective", "profiles"],
        horizontal_spacing=0.09,
    )

    colors = sample_colorscale("Viridis", np.linspace(0, 1, max(n, 2))[:n])

    profile_of = []
    for j, curve in enumerate(curves):
        fig.add_trace(
            go.Scatter(
                x=us, y=curve, mode="lines",
                line=dict(color=colors[j], width=1.6),
                opacity=0.75,
                name=f"#{it[j]}",
                legendgroup="profiles", showlegend=False,
                hovertemplate=f"#{it[j]}  y={float(ys[j]):.4f}<br>u=%{{x:.3f}}<br>%{{y:.4f}}<extra></extra>",
                visible=j == 0,
            ),
            row=1, col=2,
        )
        profile_of.append(j)

    step_of = []
    for step in range(n):
        m = step + 1
        vis = step == 0
        b = int(i_best_at[step])
        fig.add_trace(
            go.Scatter(
                x=it[:m], y=ys[:m], mode="markers+lines",
                line=dict(color="#9ca3af", width=1),
                marker=dict(size=7, color="#9ca3af", line=dict(color="white", width=1)),
                name="observed", legendgroup="observed", showlegend=True,
                hovertemplate="#%{x}  y=%{y:.4f}<extra></extra>",
                visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=it[:m], y=running_best[:m], mode="lines",
                line=dict(color="#2563eb", width=3),
                name="best so far", legendgroup="best", showlegend=True,
                hovertemplate="after %{x}: %{y:.4f}<extra></extra>",
                visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[int(it[b])], y=[float(ys[b])], mode="markers",
                marker=dict(size=15, color="#dc2626", symbol="star",
                            line=dict(color="white", width=1)),
                name="incumbent", legendgroup="incumbent", showlegend=True,
                hovertemplate=f"incumbent #{it[b]}  y={float(ys[b]):.4f}<extra></extra>",
                visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=us, y=curves[b], mode="lines",
                line=dict(color="#dc2626", width=3.5),
                name="incumbent profile", legendgroup="incumbent", showlegend=False,
                hovertemplate=f"incumbent #{it[b]}<br>u=%{{x:.3f}}<br>%{{y:.4f}}<extra></extra>",
                visible=vis,
            ),
            row=1, col=2,
        )
        step_of.extend([step] * 4)

    profile_of, step_of = np.array(profile_of), np.array(step_of)
    steps = []
    for step in range(n):
        visible = np.concatenate([profile_of <= step, step_of == step]).tolist()
        b = int(i_best_at[step])
        steps.append(
            dict(
                method="update",
                label=str(it[step]),
                args=[
                    {"visible": visible},
                    {"title.text": (
                        f"{exp_name} — through evaluation #{it[step]} "
                        f"({step + 1} of {n}), best #{it[b]} at y={float(ys[b]):.4f}"
                    )},
                ],
            )
        )

    if band:
        fig.add_vrect(x0=REACHABLE[0], x1=REACHABLE[1], fillcolor="#94a3b8",
                      opacity=0.15, line_width=0, layer="below", row=1, col=2)

    pad = 0.06 * max(float(np.ptp(ys)), 1e-9)
    curve_pad = 0.06 * max(float(np.ptp(curves)), 1e-9)
    fig.update_layout(
        title=dict(text=steps[0]["args"][1]["title.text"]),
        sliders=[dict(active=0, currentvalue=dict(prefix="through evaluation #"),
                      pad=dict(t=60), steps=steps)],
        template="plotly_white",
        width=1280,
        height=580,
        legend=dict(orientation="h", y=1.19, x=0),
        margin=dict(t=140, b=110),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="evaluation", range=[it.min() - 0.5, it.max() + 0.5],
                     dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="objective",
                     range=[float(ys.min()) - pad, float(ys.max()) + pad], row=1, col=1)
    fig.update_xaxes(title_text="u" + (" (shaded = reachable)" if band else ""),
                     range=[0, 1], row=1, col=2)
    fig.update_yaxes(title_text="shape, unnormalized",
                     range=[float(curves.min()) - curve_pad, float(curves.max()) + curve_pad],
                     row=1, col=2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument(
        "--out", default=None, help="output HTML (default: <exp-dir>_evolution.html)"
    )
    parser.add_argument("--resolution", type=int, default=300, help="points per profile")
    parser.add_argument(
        "--minimize", action="store_true",
        help="objective was minimized (default: maximized, matching bo_step.py)",
    )
    parser.add_argument(
        "--no-band", action="store_true",
        help="hide the reachable-u band (it describes the hardware, not a mock target)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="read the experiment folder on this machine instead of the Pi",
    )
    args = parser.parse_args()

    if args.local:
        experiment_io.use_local_storage()

    indices, fs, ys = load_completed(args.exp_dir)
    n = len(ys)
    if n == 0:
        raise SystemExit(f"No completed (config, run) pairs in {args.exp_dir}")

    if args.minimize:
        running_best = np.minimum.accumulate(ys)
        i_best_at = np.array([int(np.argmin(ys[: m + 1])) for m in range(n)])
    else:
        running_best = np.maximum.accumulate(ys)
        i_best_at = np.array([int(np.argmax(ys[: m + 1])) for m in range(n)])

    grid = experiment_io.profile_grid(args.resolution)
    us = np.asarray(grid).squeeze(-1)
    curves = np.stack([np.asarray(f.sample(grid)) for f in fs])

    exp_name = os.path.basename(args.exp_dir.rstrip("/"))
    fig = build_figure(indices, us, curves, ys, running_best, i_best_at, exp_name,
                       not args.no_band)

    out = args.out or f"{exp_name}_evolution.html"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")

    print(f"n={n}")
    for i, f, y in zip(indices, fs, ys):
        mark = " <-- best" if i == indices[int(i_best_at[-1])] else ""
        print(f"  {i:2d}  k={len(f.a)}  rho={float(f.rho[0]):.3f}  y={y: .4f}{mark}")
    print("wrote", out)


if __name__ == "__main__":
    main()
