"""One-shot orchestrator: run every experiment, then render every figure.

Use this for a full refresh. Day-to-day, run a single ``exp_*.py`` to refresh
one data file, or a single ``plot_*.py`` to restyle figures without re-running
the simulation.

Run:  python run_all.py [--days 2] [--seed 42] [--plot-only]
"""

from __future__ import annotations
import argparse

import bootstrap  # noqa: F401  (configures sys.path for the exp/ and plot/ scripts)

import exp_offline
import exp_online
import plot_offline
import plot_online


def main(num_days=2, seed=42, plot_only=False):
    if not plot_only:
        print("=== running offline experiment ===")
        exp_offline.main(seed)
        print("=== running online experiment ===")
        exp_online.main(num_days, seed)

    print("=== plotting offline figures ===")
    plot_offline.main()
    print("=== plotting online figures ===")
    plot_online.main()
    print("all done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all experiments and plots")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="skip simulation, only re-render figures from existing results/",
    )
    args = p.parse_args()
    main(args.days, args.seed, args.plot_only)
