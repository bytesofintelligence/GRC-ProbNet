"""
Aggregate per_structure_uncertainty.csv across all samples.

Produces:
  - class_entropy_summary.csv  (mean/std entropy per anatomical class)
  - class_entropy_bar.png      (bar chart: mean ± std)
  - class_entropy_boxplot.png  (boxplot of per-sample entropy per class)
  - Console ranking from highest to lowest uncertainty.

Usage:
  python aggregate_uncertainty.py
  python aggregate_uncertainty.py --csv <path/to/per_structure_uncertainty.csv> --out_dir <dir>
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# I/O

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_csv(rows: list[dict], fieldnames: list[str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Aggregation
def aggregate(rows: list[dict]) -> dict[str, list[float]]:
    """Return {class_name: [entropy_values_across_samples]}."""
    per_class: dict[str, list[float]] = {}
    for row in rows:
        name = row["class_name"]
        val = row["mean_entropy"]
        if val in ("", "nan"):
            continue
        per_class.setdefault(name, []).append(float(val))
    return per_class


def class_order(per_class: dict) -> list[str]:
    # Use insertion order from CSV (already sorted by class_id)
    return list(per_class.keys())


# Plots

def plot_bar(names: list[str], means: list[float], stds: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#4C72B0", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Mean Binary Class Entropy (nats)", fontsize=11)
    ax.set_title("Mean Binary Class Entropy per Anatomical Class (± std across samples)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def plot_boxplot(names: list[str], distributions: list[list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        distributions,
        labels=names,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Binary Class Entropy (nats)", fontsize=11)
    ax.set_title("Binary Class Entropy Distribution per Anatomical Class (across samples)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# Main


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-structure uncertainty across samples.")
    parser.add_argument(
        "--csv",
        default="output/mm-whs/full-stn/uncertainty_analysis/metrics/per_structure_uncertainty.csv",
        help="Path to per_structure_uncertainty.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="output/mm-whs/full-stn/uncertainty_analysis/metrics",
        help="Directory to write outputs.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {csv_path}")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} rows loaded ({len({r['sample_id'] for r in rows})} samples).")

    per_class = aggregate(rows)
    names = class_order(per_class)

    means = [float(np.mean(per_class[n])) for n in names]
    stds  = [float(np.std(per_class[n], ddof=1)) if len(per_class[n]) > 1 else 0.0 for n in names]

    
    # Summary CSV
    summary_rows = [
        {
            "class_name": n,
            "n_samples": len(per_class[n]),
            "mean_entropy": f"{means[i]:.6f}",
            "std_entropy":  f"{stds[i]:.6f}",
        }
        for i, n in enumerate(names)
    ]
    summary_csv = out_dir / "class_entropy_summary.csv"
    write_csv(summary_rows, ["class_name", "n_samples", "mean_entropy", "std_entropy"], summary_csv)
    print(f"\nSaved: {summary_csv}")

    
    # Plots
    bar_path = out_dir / "class_entropy_bar.png"
    box_path = out_dir / "class_entropy_boxplot.png"
    plot_bar(names, means, stds, bar_path)
    plot_boxplot(names, [per_class[n] for n in names], box_path)
    print(f"Saved: {bar_path}")
    print(f"Saved: {box_path}")

    
    # Ranking
    ranked = sorted(zip(names, means, stds), key=lambda x: x[1], reverse=True)
    print("\nClasses ranked by mean entropy (highest → lowest):")
    print(f"  {'Rank':<5} {'Class':<20} {'Mean H':>8}  {'Std H':>8}")
    print("  " + "-" * 45)
    for rank, (name, mean, std) in enumerate(ranked, 1):
        print(f"  {rank:<5} {name:<20} {mean:>8.4f}  {std:>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
