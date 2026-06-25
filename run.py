"""
Run:  python run.py [--days 2] [--seed 42] [--plot-only]
"""

import argparse

from exp import exp_offline, exp_online, exp_ablation
from plot import plot_offline, plot_online, plot_ablation


def offline(seed, plot_only):
    if not plot_only:
        print("=== running offline experiment ===")
        exp_offline.main(seed)
    print("=== plotting offline figures ===")
    plot_offline.main()


def online(num_days, seed, plot_only):
    if not plot_only:
        print("=== running online experiment ===")
        exp_online.main(num_days, seed)
    print("=== plotting online figures ===")
    plot_online.main()


def ablation(plot_only):
    if not plot_only:
        print("=== running ablation experiment ===")
        exp_ablation.main()
    print("=== plotting ablation figures ===")
    plot_ablation.main()


def main(num_days=2, seed=42, plot_only=False):
    offline(seed, plot_only)
    online(num_days, seed, plot_only)
    ablation(plot_only)
    print("all done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all experiments and plots")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    main(args.days, args.seed, args.plot_only)
