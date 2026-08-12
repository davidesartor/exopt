
import argparse
import os
import re

import jax.numpy as jnp
import pandas as pd

from src import experiment_io, sine


def summarize_run(exp_dir: str) -> dict:
    mode = experiment_io.experiment_mode(exp_dir)
    if mode == "functional":
        fs, ys = experiment_io.load_functional_dataset(exp_dir)
        extra = dict(
            k_final=int(len(fs[-1].x)),
            rho_best=float(fs[int(jnp.argmin(ys))].rho[0]),
        )
    else:
        xs, ys = experiment_io.load_dataset(exp_dir)
        best_x = xs[int(jnp.argmin(ys))]
        extra = dict(
            k_final=float("nan"),
            rho_best=float("nan"),
            **{f"{n}_best": float(v) for n, v in zip(sine.PARAM_NAMES, best_x)},
        )

    return dict(
        mode=mode,
        evaluations=int(len(ys)),
        best=float(ys.min()),
        final=float(ys[-1]),
        best_at=int(jnp.argmin(ys)) + 1,
        **extra,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="folder holding run folders")
    parser.add_argument("--out", help="CSV to write (default: print only)")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="read the run folders from the Pi instead of this machine",
    )
    args = parser.parse_args()

    if not args.remote:
        experiment_io.use_local_storage()

    rows = []
    for name, exp_dir in experiment_io.subdirectories(args.root):
        try:
            if experiment_io.experiment_mode(exp_dir) is None:
                continue
        except ValueError as e:
            print(f"Skipping {name}: {e}")
            continue

        m = re.match(r"(.+)_s(\d+)$", name)
        label, seed = (m.group(1), int(m.group(2))) if m else (name, -1)
        rows.append(dict(label=label, seed=seed, **summarize_run(exp_dir)))

    if not rows:
        raise SystemExit(f"No run folders found under {args.root}")

    df = pd.DataFrame(rows).sort_values(["label", "seed"])
    print(df.to_string(index=False))

    print("\nper-label summary of best objective:")
    print(
        df.groupby("label")["best"]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .to_string()
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
