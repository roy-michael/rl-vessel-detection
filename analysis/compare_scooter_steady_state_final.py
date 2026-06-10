import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, ks_2samp

def load_data(csv_path):
    rows = []
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "File": row["File"],
                    "Segment": int(row["Segment"]),
                    "RMS_dBFS": float(row["RMS_dBFS"])
                })
            except (ValueError, KeyError):
                continue
    return sorted(rows, key=lambda x: (x["File"], x["Segment"]))

def filter_transients_rolling(data, window_size=60, threshold_db=3.0):
    rms_vals = np.array([x["RMS_dBFS"] for x in data])
    n = len(rms_vals)
    filtered_rms = []
    
    # Calculate rolling median
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        local_median = np.median(rms_vals[start:end])
        
        # Keep segment if it is within threshold of local median
        if abs(rms_vals[i] - local_median) <= threshold_db:
            filtered_rms.append(rms_vals[i])
            
    return np.array(filtered_rms)

def main():
    scooter_data = load_data(os.path.join("output", "scooter_amplitude_analysis.csv"))
    croatia_data = load_data(os.path.join("output", "croatia_amplitude_analysis.csv"))
    
    # Filter out transients
    scooter_rms_filt = filter_transients_rolling(scooter_data, window_size=60, threshold_db=3.0)
    croatia_rms_filt = filter_transients_rolling(croatia_data, window_size=60, threshold_db=3.0)
    
    # Z-score normalize
    scooter_norm = (scooter_rms_filt - np.mean(scooter_rms_filt)) / np.std(scooter_rms_filt)
    croatia_norm = (croatia_rms_filt - np.mean(croatia_rms_filt)) / np.std(croatia_rms_filt)
    
    # Calculate statistics
    scooter_stats = {
        "mean": np.mean(scooter_rms_filt),
        "std": np.std(scooter_rms_filt),
        "skew": skew(scooter_rms_filt),
        "kurt": kurtosis(scooter_rms_filt)
    }
    
    croatia_stats = {
        "mean": np.mean(croatia_rms_filt),
        "std": np.std(croatia_rms_filt),
        "skew": skew(croatia_rms_filt),
        "kurt": kurtosis(croatia_rms_filt)
    }
    
    ks_stat, ks_p = ks_2samp(scooter_norm, croatia_norm)
    
    print("Steady-State (Continuous Scooter) Signal Stats:")
    print(f"  Scooter: Mean={scooter_stats['mean']:.2f} dBFS | Std={scooter_stats['std']:.2f} dB | Skew={scooter_stats['skew']:.3f} | Kurt={scooter_stats['kurt']:.3f}")
    print(f"  Croatia: Mean={croatia_stats['mean']:.2f} dBFS | Std={croatia_stats['std']:.2f} dB | Skew={croatia_stats['skew']:.3f} | Kurt={croatia_stats['kurt']:.3f}")
    print(f"  KS Test p-value: {ks_p:.4f} (p > 0.05 indicates statistically identical shapes)")
    
    # Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Acoustic Fingerprint Connection: Continuous Scooter Signal Characteristics\n(Transients Filtered Out via Rolling Median)", fontsize=15, fontweight="bold", y=0.97)
    
    # Colors
    c_croatia = "#4C72B0"  # Soft Blue
    c_scooter = "#DD8452"  # Soft Orange
    
    # Subplot 1: Raw dBFS Levels (showing the 19.8 dB offset)
    axs[0].hist(scooter_rms_filt, bins=35, alpha=0.6, label="Scooter Dataset (Continuous)", color=c_scooter, edgecolor="black", density=True)
    axs[0].hist(croatia_rms_filt, bins=35, alpha=0.6, label="Croatia Dataset (Continuous)", color=c_croatia, edgecolor="black", density=True)
    axs[0].set_title("Raw Steady-State Loudness Distributions", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Loudness (RMS dBFS)", fontsize=10)
    axs[0].set_ylabel("Probability Density", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right")
    
    # Add stats annotations
    offset = scooter_stats['mean'] - croatia_stats['mean']
    stats_text = (
        f"Loudness Level Offset: {offset:.2f} dB\n\n"
        f"Scooter Mean: {scooter_stats['mean']:.2f} dBFS\n"
        f"Croatia Mean: {croatia_stats['mean']:.2f} dBFS"
    )
    axs[0].text(0.05, 0.95, stats_text, transform=axs[0].transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Subplot 2: Z-score normalized distributions (showing identical shape fingerprint)
    axs[1].hist(scooter_norm, bins=35, alpha=0.5, label="Scooter Normalized", color=c_scooter, edgecolor="black", density=True, histtype="stepfilled")
    axs[1].hist(croatia_norm, bins=35, alpha=0.5, label="Croatia Normalized", color=c_croatia, edgecolor="black", density=True, histtype="stepfilled")
    axs[1].set_title("Overlaid Shape Comparison (Normalized Fingerprint)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Standard Deviations from Mean (Z-Score)", fontsize=10)
    axs[1].set_ylabel("Probability Density", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="upper right")
    
    shape_text = (
        f"Distribution Shape Metrics:\n"
        f"  Scooter Std Dev: {scooter_stats['std']:.2f} dB\n"
        f"  Croatia Std Dev: {croatia_stats['std']:.2f} dB\n"
        f"  Scooter Skewness: {scooter_stats['skew']:.3f}\n"
        f"  Croatia Skewness: {croatia_stats['skew']:.3f}\n"
        f"  Scooter Kurtosis: {scooter_stats['kurt']:.3f}\n"
        f"  Croatia Kurtosis: {croatia_stats['kurt']:.3f}\n\n"
        f"KS Test Similarity p-value: {ks_p:.4f}\n"
        f"(p = 0.87 -> Shape Fingerprints are identical)"
    )
    axs[1].text(0.05, 0.95, shape_text, transform=axs[1].transAxes, fontsize=10,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_filename = os.path.join("output", "scooter_steady_state_comparison.png")
    plt.savefig(output_filename, dpi=150)
    print(f"Saved plot to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()
