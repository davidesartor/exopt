
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jaxtyping import Array, Float
from plotly.subplots import make_subplots

from src import acquisition, designs, experiment_io, gp

jax.config.update("jax_enable_x64", True)


def infer_domain(xs: Float[Array, "n d"]) -> tuple[float, float]:
    lo, hi = float(jnp.min(xs)), float(jnp.max(xs))
    for candidate in (designs.UNIT, designs.VECTOR_DOMAIN):
        if candidate[0] - 1e-9 <= lo and hi <= candidate[1] + 1e-9:
            return candidate
    pad = 0.05 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def posterior_grid(
    surrogate_model: gp.GaussianProcess,
    dims: tuple[int, int],
    anchor: Float[Array, "d"],
    domain: tuple[float, float],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d0, d1 = dims
    g = jnp.linspace(*domain, resolution)
    gx, gy = jnp.meshgrid(g, g)
    pts = (
        jnp.broadcast_to(anchor, (gx.size, anchor.shape[0]))
        .at[:, d0]
        .set(gx.ravel())
        .at[:, d1]
        .set(gy.ravel())
    )
    y_best = surrogate_model.observed_ys.min()

    @jax.jit
    def moments_at(x: Float[Array, "d"]):
        mean, cov = surrogate_model.predict(x[None, :])
        mean, sd = mean.squeeze(), jnp.sqrt(jnp.clip(cov.squeeze(), 1e-300))
        return mean, sd, acquisition.log_expected_improvement(mean, sd, y_best)

    mean, sd, logei = jax.vmap(moments_at)(pts)
    shape = gx.shape
    return (
        np.asarray(g),
        np.asarray(mean).reshape(shape),
        np.asarray(sd).reshape(shape),
        np.asarray(logei).reshape(shape),
    )


def build_figure(frames, xs, ys, dims, domain, exp_name, n_obs):
    d0, d1 = dims
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=["posterior mean (objective)", "log expected improvement"],
        horizontal_spacing=0.14,
    )

    means = np.stack([f["mean"] for f in frames])
    zlo = min(float(means.min()), float(ys.min()))
    zhi = max(float(means.max()), float(ys.max()))

    trace_step = []
    for step, f in enumerate(frames):
        m, g = f["m"], f["grid"]
        seen_x, seen_y, seen_obj = (
            np.asarray(xs[:m, d0]),
            np.asarray(xs[:m, d1]),
            np.asarray(ys[:m]),
        )
        vis = step == 0
        nxt = f["next"]
        nx = [float(nxt[d0])] if nxt is not None else []
        ny = [float(nxt[d1])] if nxt is not None else []

        fig.add_trace(
            go.Surface(
                x=g, y=g, z=f["mean"], surfacecolor=f["mean"],
                colorscale="Viridis", cmin=zlo, cmax=zhi,
                opacity=0.92,
                customdata=f["sd"],
                colorbar=dict(x=0.485, len=0.72, thickness=11,
                              title=dict(text="objective", side="right")),
                contours=dict(z=dict(show=True, usecolormap=True, project_z=False,
                                     width=1, highlightwidth=2)),
                hovertemplate=(f"x{d0}=%{{x:.3f}}<br>x{d1}=%{{y:.3f}}"
                               "<br>mean=%{z:.4f}<br>sd=%{customdata:.4f}<extra></extra>"),
                name="surrogate mean", showscale=True, visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=seen_x, y=seen_y, z=seen_obj, mode="markers",
                marker=dict(size=4, color="white", line=dict(color="black", width=1)),
                text=[f"#{k + 1} y={v:.4f}" for k, v in enumerate(seen_obj)],
                hovertemplate="%{text}<extra></extra>",
                name="observed", legendgroup="observed", visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(xs[f["best"], d0])], y=[float(xs[f["best"], d1])],
                z=[float(ys[f["best"]])], mode="markers",
                marker=dict(size=7, color="#facc15", symbol="diamond",
                            line=dict(color="black", width=1)),
                text=[f"incumbent #{f['best'] + 1} y={float(ys[f['best']]):.4f}"],
                hovertemplate="%{text}<extra></extra>",
                name="incumbent", legendgroup="incumbent", visible=vis,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=nx, y=ny, z=[f["next_mean"]] if nxt is not None else [], mode="markers",
                marker=dict(size=6, color="#f43f5e", symbol="x",
                            line=dict(color="white", width=1)),
                text=[f["next_label"]],
                hovertemplate="%{text}<extra></extra>",
                name="proposed next", legendgroup="next", visible=vis,
            ),
            row=1, col=1,
        )

        logei = f["logei"]
        hi = float(logei.max())
        fig.add_trace(
            go.Heatmap(
                x=g, y=g, z=logei,
                colorscale="Cividis", zmin=max(float(logei.min()), hi - 20.0), zmax=hi,
                colorbar=dict(x=1.015, len=0.72, thickness=11,
                              title=dict(text="log EI", side="right")),
                hovertemplate=(f"x{d0}=%{{x:.3f}}<br>x{d1}=%{{y:.3f}}"
                               "<br>log EI=%{z:.3f}<extra></extra>"),
                showscale=True, visible=vis,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=seen_x, y=seen_y, mode="markers",
                marker=dict(size=7, color="white", line=dict(color="black", width=1)),
                text=[f"#{k + 1} y={v:.4f}" for k, v in enumerate(seen_obj)],
                hovertemplate="%{text}<extra></extra>",
                name="observed", legendgroup="observed", showlegend=False, visible=vis,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=nx, y=ny, mode="markers",
                marker=dict(size=13, color="#f43f5e", symbol="x",
                            line=dict(color="white", width=1)),
                text=[f["next_label"]],
                hovertemplate="%{text}<extra></extra>",
                name="proposed next", legendgroup="next", showlegend=False, visible=vis,
            ),
            row=1, col=2,
        )
        trace_step.extend([step] * 7)

    trace_step = np.array(trace_step)
    steps = [
        dict(
            method="update",
            label=str(f["m"]),
            args=[
                {"visible": (trace_step == step).tolist()},
                {"title.text": (
                    f"{exp_name} — surrogate after {f['m']} of {n_obs} evaluations "
                    f"(lengthscale {np.array2string(np.asarray(f['rho']), precision=3)})"
                )},
            ],
        )
        for step, f in enumerate(frames)
    ]

    fig.update_layout(
        title=dict(text=steps[0]["args"][1]["title.text"]),
        sliders=[dict(active=0, currentvalue=dict(prefix="observations: "),
                      pad=dict(t=60), steps=steps)],
        template="plotly_white",
        width=1400,
        height=680,
        legend=dict(orientation="h", y=1.14, x=0),
        margin=dict(t=120, b=110, r=90),
        scene=dict(
            xaxis=dict(title=f"x{d0}", range=list(domain)),
            yaxis=dict(title=f"x{d1}", range=list(domain)),
            zaxis=dict(title="objective", range=[zlo, zhi]),
            camera=dict(eye=dict(x=1.35, y=-1.45, z=0.75)),
            aspectratio=dict(x=1, y=1, z=0.75),
            domain=dict(x=[0.0, 0.46], y=[0.0, 1.0]),
        ),
    )
    fig.update_xaxes(title_text=f"x{d0}", range=list(domain), constrain="domain", row=1, col=2)
    fig.update_yaxes(title_text=f"x{d1}", range=list(domain), scaleanchor="x2", scaleratio=1,
                     row=1, col=2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument(
        "--out", default=None, help="output HTML (default: <exp-dir>_surrogate_evolution.html)"
    )
    parser.add_argument(
        "--dims", type=int, nargs=2, default=(0, 1), metavar=("I", "J"),
        help="which two parameter axes to plot (default 0 1)",
    )
    parser.add_argument("--resolution", type=int, default=70, help="grid resolution per axis")
    parser.add_argument("--start", type=int, default=3, help="first step's number of observations")
    parser.add_argument(
        "--minimize", action="store_true",
        help="objective was minimized (default: maximized, matching bo_step.py)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="read the experiment folder on this machine instead of the Pi",
    )
    args = parser.parse_args()

    if args.local:
        experiment_io.use_local_storage()

    xs, ys = experiment_io.load_dataset(args.exp_dir)
    n = len(xs)
    if n == 0:
        raise SystemExit(f"No completed (config, run) pairs in {args.exp_dir}")
    dim = xs.shape[-1]
    d0, d1 = args.dims
    if not (0 <= d0 < dim and 0 <= d1 < dim):
        raise SystemExit(f"--dims {d0} {d1} out of range for a {dim}D problem.")
    if args.start < 2:
        raise SystemExit("--start must be at least 2; a GP cannot be fit to one point.")
    if n < args.start:
        raise SystemExit(f"Only {n} observations in {args.exp_dir}; --start is {args.start}.")

    pending = sorted(
        experiment_io.config_numbers(args.exp_dir) - experiment_io.run_numbers(args.exp_dir)
    )
    x_pending = experiment_io.read_config(args.exp_dir, pending[0]) if pending else None

    domain = infer_domain(xs)
    take_best = jnp.argmin if args.minimize else jnp.argmax
    sign = 1.0 if args.minimize else -1.0

    frames = []
    for m in range(args.start, n + 1):
        model = gp.GaussianProcess().fit(xs[:m], sign * ys[:m])
        best = int(take_best(ys[:m]))
        g, mean, sd, logei = posterior_grid(
            model, (d0, d1), xs[best], domain, args.resolution
        )
        if m < n:
            nxt, label = xs[m], f"proposed #{m + 1}, measured y={float(ys[m]):.4f}"
        else:
            nxt = x_pending
            label = f"pending: config_{pending[0]}" if pending else ""
        next_mean = None
        if nxt is not None:
            predicted, _ = model.predict(jnp.asarray(nxt)[None, :])
            next_mean = sign * float(predicted.squeeze())
            label += f"<br>predicted mean {next_mean:.4f}"
        frames.append(
            dict(m=m, best=best, grid=g, mean=sign * mean, sd=sd, logei=logei,
                 rho=model.rho, next=nxt, next_label=label, next_mean=next_mean)
        )
        print(f"  fit on {m:2d} obs  rho={np.asarray(model.rho)}  nugget={float(model.g):.3g}")

    exp_name = os.path.basename(args.exp_dir.rstrip("/"))
    fig = build_figure(frames, xs, ys, (d0, d1), domain, exp_name, n)

    out = args.out or f"{exp_name}_surrogate_evolution.html"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print("wrote", out)


if __name__ == "__main__":
    main()
