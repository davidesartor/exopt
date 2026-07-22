"""Plot a heatmap of the surrogate model mean over the domain, with the explored
points overlaid, and save it to a self-contained HTML file (plotly)."""

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from jaxtyping import Array, Float

from src import designs, experiment_io, gp

jax.config.update("jax_enable_x64", True)


def surrogate_mean_grid(
    surrogate_model: gp.GaussianProcess,
    dims: tuple[int, int],
    anchor: Float[Array, "d"],
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the surrogate posterior mean over a 2D slice of the domain.

    The two axes in ``dims`` sweep the search domain; every other dimension is
    held fixed at ``anchor`` (typically the incumbent best config).
    """
    d0, d1 = dims
    g = jnp.linspace(*designs.VECTOR_DOMAIN, resolution)
    gx, gy = jnp.meshgrid(g, g)

    pts = jnp.broadcast_to(anchor, (gx.size, anchor.shape[0])).at[:, d0].set(
        gx.ravel()
    ).at[:, d1].set(gy.ravel())

    @jax.jit
    def mean_at(x: Float[Array, "d"]) -> Float[Array, ""]:
        mean, _ = surrogate_model.predict(x[None, :])
        return mean.squeeze()

    z = jax.vmap(mean_at)(pts).reshape(gx.shape)
    return np.asarray(g), np.asarray(g), np.asarray(z)


def plot(
    exp_dir: str,
    output: str,
    dims: tuple[int, int] = (0, 1),
    resolution: int = 150,
) -> str:
    xs, ys = experiment_io.load_dataset(exp_dir)
    if len(xs) == 0:
        raise SystemExit(f"No completed (config, run) pairs in {exp_dir}.")

    dim = xs.shape[-1]
    d0, d1 = dims
    if not (0 <= d0 < dim and 0 <= d1 < dim):
        raise SystemExit(f"--dims {d0} {d1} out of range for a {dim}D problem.")

    surrogate_model = gp.GaussianProcess().fit(xs, ys)
    best = int(jnp.argmin(ys))
    anchor = xs[best]

    gx, gy, z = surrogate_mean_grid(surrogate_model, (d0, d1), anchor, resolution)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=gx,
            y=gy,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="surrogate mean"),
            opacity=0.9,
            hovertemplate=f"x{d0}=%{{x:.3f}}<br>x{d1}=%{{y:.3f}}<br>mean=%{{z:.4f}}<extra></extra>",
        )
    )

    # explored points sit at their observed objective, colored by that value
    fig.add_trace(
        go.Scatter3d(
            x=np.asarray(xs[:, d0]),
            y=np.asarray(xs[:, d1]),
            z=np.asarray(ys),
            mode="markers",
            marker=dict(
                size=5,
                color=np.asarray(ys),
                colorscale="Cividis",
                line=dict(color="white", width=1),
                showscale=False,
            ),
            text=[f"y={float(v):.4f}" for v in ys],
            hovertemplate="explored<br>x%d=%%{x:.3f}<br>x%d=%%{y:.3f}<br>%%{text}<extra></extra>"
            % (d0, d1),
            name="explored",
        )
    )

    # incumbent best, a large diamond so identity does not rest on color alone
    fig.add_trace(
        go.Scatter3d(
            x=[float(xs[best, d0])],
            y=[float(xs[best, d1])],
            z=[float(ys[best])],
            mode="markers",
            marker=dict(size=9, color="#DD3333", symbol="diamond", line=dict(color="white", width=1)),
            text=[f"best y={float(ys[best]):.4f}"],
            hovertemplate="%{text}<extra></extra>",
            name="best",
        )
    )

    slice_note = ""
    if dim > 2:
        held = ", ".join(
            f"x{k}={float(anchor[k]):.3f}" for k in range(dim) if k not in (d0, d1)
        )
        slice_note = f"  (slice at best: {held})"

    fig.update_layout(
        title=f"Surrogate mean over (x{d0}, x{d1}){slice_note} — {len(xs)} evaluations",
        scene=dict(
            xaxis=dict(title=f"x{d0}", range=[0, 1]),
            yaxis=dict(title=f"x{d1}", range=[0, 1]),
            zaxis=dict(title="objective"),
        ),
        template="plotly_white",
        width=820,
        height=740,
    )

    fig.write_html(output, include_plotlyjs="cdn")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment (results) folder")
    parser.add_argument(
        "--output", default=None, help="output HTML path (default: <exp-dir>_surrogate.html)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("I", "J"),
        help="which two parameter axes to plot (default 0 1)",
    )
    parser.add_argument("--resolution", type=int, default=150, help="grid resolution")
    parser.add_argument(
        "--local",
        action="store_true",
        help="read/write the experiment folder on this machine instead of the Pi",
    )
    args = parser.parse_args()

    if args.local:
        experiment_io.use_local_storage()

    output = args.output or f"{os.path.basename(args.exp_dir.rstrip('/'))}_surrogate.html"
    path = plot(args.exp_dir, output, tuple(args.dims), args.resolution)
    print(f"Saved surrogate heatmap to {path}")


if __name__ == "__main__":
    main()
