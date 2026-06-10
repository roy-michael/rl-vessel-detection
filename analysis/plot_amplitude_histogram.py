import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Canonical input paths (both datasets loaded together for comparison)
SCOOTER_CSV = os.path.join("output", "scooter_amplitude_analysis.csv")
CROATIA_CSV = os.path.join("output", "croatia_amplitude_analysis.csv")
OUTPUT_IMG  = os.path.join("output", "amplitude_histograms.png")

C_SCOOTER = "#DD8452"   # warm orange
C_CROATIA = "#4C72B0"   # slate blue


def load_amplitude_data(csv_path):
    """Reads a CSV and returns arrays of (Peak_dBFS, RMS_dBFS)."""
    peak_db, rms_db = [], []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                peak_db.append(float(row["Peak_dBFS"]))
                rms_db.append(float(row["RMS_dBFS"]))
            except (ValueError, KeyError):
                continue
    return np.array(peak_db), np.array(rms_db)


def add_stats(ax, data, color, label):
    """Overlay a vertical mean line and stats box."""
    mean_val = np.mean(data)
    ax.axvline(mean_val, color=color, linestyle="--", linewidth=1.4,
               label=f"{label} mean: {mean_val:.1f} dBFS")


def main():
    print(f"Loading {SCOOTER_CSV} ...")
    sc_peak, sc_rms = load_amplitude_data(SCOOTER_CSV)
    print(f"  {len(sc_rms)} segments loaded.")

    print(f"Loading {CROATIA_CSV} ...")
    cr_peak, cr_rms = load_amplitude_data(CROATIA_CSV)
    print(f"  {len(cr_rms)} segments loaded.")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "Amplitude Distribution Comparison — Scooter Dataset vs Croatia Ocean Sonics",
        fontsize=14, fontweight="bold", y=1.01
    )

    bins = 45

    # ---------- Left: RMS dBFS ----------
    ax = axs[0]
    sc_rms_f = sc_rms[sc_rms > -99]
    cr_rms_f = cr_rms[cr_rms > -99]
    ax.hist(sc_rms_f, bins=bins, alpha=0.6, color=C_SCOOTER,
            edgecolor="white", linewidth=0.4, density=True, label="Scooter")
    ax.hist(cr_rms_f, bins=bins, alpha=0.6, color=C_CROATIA,
            edgecolor="white", linewidth=0.4, density=True, label="Croatia")
    add_stats(ax, sc_rms_f, C_SCOOTER, "Scooter")
    add_stats(ax, cr_rms_f, C_CROATIA, "Croatia")

    offset = np.mean(sc_rms_f) - np.mean(cr_rms_f)
    ax.set_title("RMS Amplitude (dBFS)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Loudness (dBFS)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97,
            f"Scooter mean: {np.mean(sc_rms_f):.1f} dBFS\n"
            f"Croatia mean: {np.mean(cr_rms_f):.1f} dBFS\n"
            f"Offset: {offset:+.1f} dB\n"
            f"Scooter std: {np.std(sc_rms_f):.2f} dB\n"
            f"Croatia std: {np.std(cr_rms_f):.2f} dB",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    # ---------- Right: Peak dBFS ----------
    ax = axs[1]
    sc_pk_f = sc_peak[sc_peak > -99]
    cr_pk_f = cr_peak[cr_peak > -99]
    ax.hist(sc_pk_f, bins=bins, alpha=0.6, color=C_SCOOTER,
            edgecolor="white", linewidth=0.4, density=True, label="Scooter")
    ax.hist(cr_pk_f, bins=bins, alpha=0.6, color=C_CROATIA,
            edgecolor="white", linewidth=0.4, density=True, label="Croatia")
    add_stats(ax, sc_pk_f, C_SCOOTER, "Scooter")
    add_stats(ax, cr_pk_f, C_CROATIA, "Croatia")

    ax.set_title("Peak Amplitude (dBFS)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Loudness (dBFS)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97,
            f"Scooter mean: {np.mean(sc_pk_f):.1f} dBFS\n"
            f"Croatia mean: {np.mean(cr_pk_f):.1f} dBFS\n"
            f"Scooter std: {np.std(sc_pk_f):.2f} dB\n"
            f"Croatia std: {np.std(cr_pk_f):.2f} dB",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {os.path.abspath(OUTPUT_IMG)}")


if __name__ == "__main__":
    main()
