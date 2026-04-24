#!/usr/bin/env python3
"""
Plot before/after reward comparison from two eval JSON files.

Usage (after training/baseline_eval.py runs for both pre- and post-training):
    python plot_rewards.py --before baseline_results.json --after trained_results.json
"""
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="baseline_results.json")
    p.add_argument("--after", default="trained_results.json")
    p.add_argument("--outdir", default="plots")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    pre = json.loads(Path(args.before).read_text())
    post = json.loads(Path(args.after).read_text())

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    BEFORE_COLOR = "#d9534f"
    AFTER_COLOR = "#5cb85c"

    # Plot 1: Win rate bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("TICKETMELT — Win Rate Before vs After GRPO", fontsize=13, fontweight="bold")
    values = [pre["win_rate"], post["win_rate"]]
    bars = ax.bar(["Before", "After"], values, color=[BEFORE_COLOR, AFTER_COLOR], width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Win Rate")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    plt.tight_layout()
    out1 = outdir / "reward_curve.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out1}")

    # Plot 2: Per-component rewards
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("TICKETMELT — Reward Components Before vs After GRPO", fontsize=13, fontweight="bold")
    components = ["avg_r1", "avg_r2", "avg_r3", "avg_r4"]
    labels = ["R1\n(own svc)", "R2\n(site uptime)", "R3\n(clean deploys)", "R4\n(yield)"]
    x = list(range(len(components)))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [pre[c] for c in components], w,
           label="Before", color=BEFORE_COLOR, alpha=0.85)
    ax.bar([i + w / 2 for i in x], [post[c] for c in components], w,
           label="After", color=AFTER_COLOR, alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Component Score (0-1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    plt.tight_layout()
    out2 = outdir / "component_rewards.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")


if __name__ == "__main__":
    main()
