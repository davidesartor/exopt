from jaxtyping import Array, Float, Scalar
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp

from src import experiment_io, gp

jax.config.update("jax_enable_x64", True)


def propose_next(
    xs: Float[Array, "n d"],
    ys: Float[Array, "n"],
    seed: int,
    acquisition_raw_samples: int = 256,
    acquisition_max_restarts: int = 5,
) -> tuple[Float[Array, "d"], gp.GaussianProcess]:
    dim = xs.shape[-1]
    surrogate_model = gp.GaussianProcess().fit(xs, ys)

    @jax.jit
    @jax.value_and_grad
    def acquisition_loss(x: Float[Array, "d"]) -> Scalar:
        mu, cov = surrogate_model.predict(x[None, :])
        return -gp.log_expected_improvement(
            mu=mu.squeeze(),
            sigma=cov.squeeze() ** 0.5,
            y_best=surrogate_model.observed_ys.min(),
        )

    # restart L-BFGS-B from the best of a Latin hypercube of initial candidates
    candidates = sp.stats.qmc.LatinHypercube(d=dim, rng=seed).random(
        n=acquisition_raw_samples
    )
    losses = [acquisition_loss(c)[0] for c in candidates]
    candidates = candidates[np.argsort(losses)[:acquisition_max_restarts]]

    results = [
        sp.optimize.minimize(
            fun=acquisition_loss,
            x0=c,
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * dim,
            options=dict(maxiter=100, ftol=gp.EPS, gtol=0.0),
        )
        for c in candidates
    ]

    # return the location of the best local optimum found and the fitted model
    losses = jnp.array([result.fun for result in results])
    x_next = jnp.array(results[jnp.argmin(losses)].x)
    return x_next, surrogate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-samples", type=int, default=1024)
    parser.add_argument("--max-restarts", type=int, default=16)
    parser.add_argument("--minimize", action="store_true", help="minimize the objective (default: maximize)")
    args = parser.parse_args()

    xs, ys = experiment_io.load_dataset(args.exp_dir)
    if len(xs) == 0:
        raise SystemExit(
            f"No completed (config, run) pairs in {args.exp_dir}; "
            "run initial_design.py and mock_experiment.py first."
        )

    if not args.minimize:
        ys = -ys

    print(f"Loaded {len(xs)} observations, best so far: {ys.min():.6f}")
    print("Fitting surrogate and optimizing expected improvement...")
    x_next, _ = propose_next(
        xs,
        ys,
        seed=args.seed,
        acquisition_raw_samples=args.raw_samples,
        acquisition_max_restarts=args.max_restarts,
    )

    i = experiment_io.next_index(args.exp_dir)
    experiment_io.save_config(args.exp_dir, i, x_next)
    print(f"Proposed next point -> config_{i}.txt: {x_next}")


if __name__ == "__main__":
    main()
