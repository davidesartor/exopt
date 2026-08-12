
import argparse
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import experiment_io

REACHABLE = (0.137, 0.908)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, help="experiment folder")
    parser.add_argument("--out", default="results/plots/evolution.png")
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="objective was minimized (default: maximized, matching bo_step.py)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="read the experiment folder on this machine instead of the Pi",
    )
    args = parser.parse_args()

    if args.local:
        experiment_io.use_local_storage()

    fs, ys = experiment_io.load_functional_dataset(args.exp_dir)
    ys = np.asarray(ys)
    if len(ys) == 0:
        raise SystemExit(f"No completed (config, run) pairs in {args.exp_dir}")
    n = len(ys)
    it = np.arange(1, n + 1)

    if args.minimize:
        best = np.minimum.accumulate(ys)
        i_best = int(np.argmin(ys))
    else:
        best = np.maximum.accumulate(ys)
        i_best = int(np.argmax(ys))

    completed = sorted(
        experiment_io.config_numbers(args.exp_dir) & experiment_io.run_numbers(args.exp_dir)
    )
    peaks = []
    for i in completed:
        with experiment_io._pi().open(f"{args.exp_dir}/run_{i}.txt") as fh:
            df = pd.read_csv(io.StringIO(fh.read().decode()), sep=" ")
        peaks.append(float(np.abs(df.torque_0.values).max()))
    peaks = np.array(peaks)

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.4))

    ax.plot(it, ys, "o-", color="#9ca3af", lw=1, ms=6, label="observed")
    ax.plot(it, best, "-", color="#2563eb", lw=2.5, label="best so far")
    ax.plot(i_best + 1, ys[i_best], "*", color="#dc2626", ms=18, label="best")
    ax.set_title(f"objective (n={n})")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("objective")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    grid = experiment_io.profile_grid(300)
    u = np.asarray(grid).squeeze(-1)
    cmap = plt.get_cmap("viridis")
    for j, f in enumerate(fs):
        is_best = j == i_best
        ax2.plot(
            u, np.asarray(f.sample(grid)),
            color="#dc2626" if is_best else cmap(j / max(n - 1, 1)),
            lw=2.5 if is_best else 1.2,
            alpha=1.0 if is_best else 0.6,
            zorder=3 if is_best else 2,
            label=f"#{j + 1} (best)" if is_best else None,
        )
    ax2.axvspan(*REACHABLE, color="#94a3b8", alpha=0.15, zorder=0)
    ax2.set_title("profiles (shaded = reachable u)")
    ax2.set_xlabel("u")
    ax2.set_ylabel("shape, unnormalized")
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=9)

    ax3.bar(it, peaks, color="#2563eb")
    ax3.set_yscale("log")
    ax3.set_title("peak |commanded torque| per run")
    ax3.set_xlabel("evaluation")
    ax3.set_ylabel("Nm (log)")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)

    print(f"n={n}")
    for j, (f, y, p) in enumerate(zip(fs, ys, peaks), start=1):
        mark = " <-- best" if j - 1 == i_best else ""
        print(f"  {j:2d}  k={len(f.a)}  rho={float(f.rho[0]):.3f}  "
              f"y={y: .4f}  peak_torque={p:8.4f} Nm{mark}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
