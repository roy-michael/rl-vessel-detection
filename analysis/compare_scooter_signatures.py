import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def load_rms_series(path):
    series = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                series.append({
                    "File": row["File"],
                    "Segment": int(row["Segment"]),
                    "RMS_dBFS": float(row["RMS_dBFS"]),
                    "Peak_dBFS": float(row["Peak_dBFS"])
                })
            except (ValueError, KeyError):
                continue
    # Sort chronologically by file name and segment
    return sorted(series, key=lambda x: (x["File"], x["Segment"]))

def min_max_normalize(v):
    v_min, v_max = np.min(v), np.max(v)
    if v_max - v_min == 0:
        return np.zeros_like(v)
    return (v - v_min) / (v_max - v_min)

def z_score_normalize(v):
    std = np.std(v)
    if std == 0:
        return np.zeros_like(v)
    return (v - np.mean(v)) / std

def main():
    scooter_data = load_rms_series(os.path.join("output", "scooter_amplitude_analysis.csv"))
    croatia_data = load_rms_series(os.path.join("output", "croatia_amplitude_analysis.csv"))
    
    # 1. Define the Croatia scooter pass event
    # We select the pass around RBW6737_20250725_084900.wav from segment 6 to 17 (12 seconds)
    croatia_pass_file = "RBW6737_20250725_084900.wav"
    croatia_pass = [x for x in croatia_data if x["File"] == croatia_pass_file and 6 <= x["Segment"] <= 17]
    croatia_pass_rms = np.array([x["RMS_dBFS"] for x in croatia_pass])
    croatia_pass_segs = [x["Segment"] for x in croatia_pass]
    
    print(f"Croatia Scooter Pass Envelope (from {croatia_pass_file}):")
    for s, val in zip(croatia_pass_segs, croatia_pass_rms):
        print(f"  Segment {s:2d}: {val:.2f} dBFS")
        
    n_template = len(croatia_pass_rms)
    
    # 2. Slide over the Scooter dataset to find the most similar envelope shape
    # We will search within each file in the Scooter dataset so we don't cross boundaries
    scooter_files = sorted(list(set(x["File"] for x in scooter_data)))
    
    best_corr = -1.0
    best_window = None
    best_file = None
    best_start_idx = -1
    
    # We want to find a window in Scooter that matches the Croatia pass shape
    for sf in scooter_files:
        sf_data = [x for x in scooter_data if x["File"] == sf]
        sf_rms = np.array([x["RMS_dBFS"] for x in sf_data])
        
        if len(sf_rms) < n_template:
            continue
            
        for i in range(len(sf_rms) - n_template + 1):
            window = sf_rms[i : i + n_template]
            
            # Check correlation
            corr = np.corrcoef(croatia_pass_rms, window)[0, 1]
            if np.isnan(corr):
                continue
                
            if corr > best_corr:
                best_corr = corr
                best_window = window
                best_file = sf
                best_start_idx = i
                
    print(f"\nBest match in Scooter dataset:")
    print(f"  File: {best_file}")
    print(f"  Segments: {best_start_idx + 1} to {best_start_idx + n_template}")
    print(f"  Pearson Correlation: {best_corr:.5f}")
    
    # Let's print the best match values
    for idx, val in enumerate(best_window):
        print(f"  Segment {best_start_idx + 1 + idx:2d}: {val:.2f} dBFS")
        
    # 3. Plot the comparison
    # Modern clean styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Acoustic Signature Comparison: Croatia vs Scooter Dataset", fontsize=15, fontweight="bold", y=0.97)
    
    # Colors
    c_croatia = "#4C72B0"  # Soft Blue
    c_scooter = "#DD8452"  # Soft Orange
    
    # Plot 1: Raw dBFS levels (showing absolute differences)
    time_axis = np.arange(n_template)
    axs[0].plot(time_axis, croatia_pass_rms, color=c_croatia, marker="o", linewidth=2.5, label="Croatia (Ocean Sonics)")
    axs[0].plot(time_axis, best_window, color=c_scooter, marker="s", linewidth=2.5, label="Scooter (RBW6922)")
    axs[0].set_title("Raw Amplitude Envelopes (dBFS)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Relative Time (Seconds)", fontsize=10)
    axs[0].set_ylabel("Loudness (dBFS)", fontsize=10)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend()
    
    # Annotate peak difference
    peak_diff = np.max(croatia_pass_rms) - np.max(best_window)
    axs[0].text(0.05, 0.05, f"Peak Amplitude Difference: {peak_diff:.1f} dB\n(Croatia is louder by ~{peak_diff:.1f} dB)",
                transform=axs[0].transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    # Plot 2: Normalized profiles (showing the shape connection)
    norm_croatia = min_max_normalize(croatia_pass_rms)
    norm_scooter = min_max_normalize(best_window)
    
    axs[1].plot(time_axis, norm_croatia, color=c_croatia, marker="o", linewidth=2.5, label="Croatia (Ocean Sonics)")
    axs[1].plot(time_axis, norm_scooter, color=c_scooter, marker="s", linewidth=2.5, label="Scooter (RBW6922)")
    axs[1].set_title("Normalized Envelope Shapes (Min-Max Scaled)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Relative Time (Seconds)", fontsize=10)
    axs[1].set_ylabel("Normalized Amplitude (0.0 to 1.0)", fontsize=10)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()
    
    # Add correlation text
    axs[1].text(0.05, 0.05, f"Shape Correlation (Pearson r): {best_corr:.4f}\nMean Squared Error (Scaled): {np.mean((norm_croatia - norm_scooter)**2):.4f}",
                transform=axs[1].transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_filename = os.path.join("output", "scooter_signature_comparison.png")
    plt.savefig(output_filename, dpi=150)
    print(f"\nSignature comparison plot saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()
