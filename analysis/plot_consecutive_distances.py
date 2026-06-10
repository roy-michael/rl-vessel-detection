"""
Reads the pre-computed segment distance CSVs for both datasets and produces
a single stacked figure showing consecutive segment transitions over time.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SCOOTER_CSV = os.path.join("output", "scooter_segment_distances.csv")
CROATIA_CSV = os.path.join("output", "croatia_segment_distances.csv")
OUTPUT_IMG  = os.path.join("output", "consecutive_distances.png")

C_SCOOTER = "#DD8452"
C_CROATIA = "#4C72B0"


def load_distances(csv_path):
    """Returns parallel arrays of (time_label, distance_to_next)."""
    times, dists = [], []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = float(row["Distance_to_Next"])
                times.append(row.get("Time", ""))
                dists.append(d)
            except (ValueError, KeyError):
                continue
    return times, np.array(dists)


def plot_panel(ax, dists, color, label, threshold=10.0):
    """Plots a single consecutive-distance timeline panel."""
    x = np.arange(len(dists))
    ax.plot(x, dists, color=color, linewidth=0.6, alpha=0.75)
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Transition threshold ({threshold} dB)")

    # Shade high-transition regions
    above = dists >= threshold
    ax.fill_between(x, 0, dists, where=above, color="crimson", alpha=0.18,
                    label="High transition")
    ax.fill_between(x, 0, dists, where=~above, color=color, alpha=0.12)

    n_trans = int(np.sum(above))
    mean_d  = float(np.mean(dists))
    ax.text(0.99, 0.97,
            f"Segments: {len(dists):,}\nMean dist: {mean_d:.2f} dB\n"
            f"Transitions ≥{threshold} dB: {n_trans}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance to Next Seg (dB)", fontsize=9)
    ax.set_xlim(0, len(dists))
    ax.set_ylim(0, max(dists.max() * 1.05, threshold * 1.5))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=8, loc="upper left")


def main():
    print(f"Loading {SCOOTER_CSV} ...")
    _, sc_dists = load_distances(SCOOTER_CSV)
    print(f"  {len(sc_dists)} segments")

    print(f"Loading {CROATIA_CSV} ...")
    _, cr_dists = load_distances(CROATIA_CSV)
    print(f"  {len(cr_dists)} segments")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(
        "Consecutive Segment Transitions — Scooter vs Croatia\n"
        "(Manhattan dB distance between adjacent 1-second segments)",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 1, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    plot_panel(ax1, sc_dists, C_SCOOTER,
               f"Scooter Dataset  ({len(sc_dists):,} segments, 42 files)")
    plot_panel(ax2, cr_dists, C_CROATIA,
               f"Croatia 2507_1  ({len(cr_dists):,} segments, 90 files)")

    ax2.set_xlabel("Segment Index", fontsize=9)

    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {os.path.abspath(OUTPUT_IMG)}")


if __name__ == "__main__":
    main()
