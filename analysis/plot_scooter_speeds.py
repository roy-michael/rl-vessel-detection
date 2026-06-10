import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_file_speed_profile(filepath, sr=16000, seg_len_sec=1.0):
    """
    Computes the 60-second speed profile (motor hum frequency in Hz) for an audio file.
    """
    y, sr_actual = librosa.load(filepath, sr=sr, mono=True)
    if len(y) < sr:
        return np.zeros(60)
        
    win_len = int(seg_len_sec * sr)
    n_wins = min(len(y) // win_len, 60)
    
    # Filter above 150 Hz to remove low-frequency mooring/grid slope
    y_hp = butter_highpass_filter(y, 150.0, sr, order=5)
    
    timeline_freqs = []
    
    for i in range(n_wins):
        start_idx = i * win_len
        end_idx = start_idx + win_len
        
        window = y_hp[start_idx:end_idx]
        std_val = np.std(window)
        if std_val < 1e-10:
            timeline_freqs.append(np.nan)
            continue
            
        windowed = window * np.hanning(len(window))
        spec = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
        mag = np.abs(spec)
        
        # Search band: 580 Hz to 950 Hz (excludes 539 Hz mooring line)
        idx_band = np.where((freqs >= 580.0) & (freqs <= 950.0))[0]
        if len(idx_band) > 0:
            max_idx = idx_band[np.argmax(mag[idx_band])]
            peak_hz = freqs[max_idx]
            timeline_freqs.append(peak_hz)
        else:
            timeline_freqs.append(np.nan)
            
    # Pad to 60 elements if shorter
    if len(timeline_freqs) < 60:
        timeline_freqs.extend([np.nan] * (60 - len(timeline_freqs)))
        
    # Interpolate missing NaNs for clean line plots
    timeline_array = np.array(timeline_freqs[:60])
    nans = np.isnan(timeline_array)
    if np.any(nans) and not np.all(nans):
        x_valid = np.where(~nans)[0]
        y_valid = timeline_array[~nans]
        timeline_array[nans] = np.interp(np.where(nans)[0], x_valid, y_valid)
    elif np.all(nans):
        timeline_array[:] = 856.0  # Fallback to high speed
        
    # Apply a median filter to remove transient noise spikes/dropouts
    from scipy.signal import medfilt
    timeline_array = medfilt(timeline_array, kernel_size=3)
        
    return timeline_array

def main():
    scooter_dir = r"C:\Users\Roy\Recordings\scooter"
    scooter_files = sorted(glob.glob(os.path.join(scooter_dir, "*.wav")))
    
    output_img = "scooter_speeds_profile.png"
    artifact_dir = "output"
    artifact_path = os.path.join(artifact_dir, output_img)
    
    print(f"Scanning and extracting speed profiles for all {len(scooter_files)} Scooter files...")
    
    all_profiles = []
    file_labels = []
    
    # Track loading progress
    for idx, f in enumerate(scooter_files):
        basename = os.path.basename(f)
        profile = get_file_speed_profile(f)
        all_profiles.append(profile)
        # Simplify label (remove prefix and extension)
        lbl = basename.replace("RBW6922_", "").replace(".wav", "")
        file_labels.append(lbl)
        print(f"  [{idx+1}/{len(scooter_files)}] Loaded {basename}")
        
    all_profiles_matrix = np.array(all_profiles)
    
    # Modern clean styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"
    plt.rcParams["xtick.color"] = "#2C3E50"
    plt.rcParams["ytick.color"] = "#2C3E50"
    
    fig = plt.figure(figsize=(18, 11), facecolor="#F8F9FA")
    fig.suptitle("Scooter Operating Speed Profiles & Transitions\nBased on Electric Motor Magnetic Pole-Passing Hum Frequency (Direct Spectral Peak Tracking, 580 - 950 Hz)", 
                 fontsize=18, fontweight="bold", y=0.98, color="#1A252C")
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.22)
    
    # --- Panel 1: Selected Timelines (Detailed Transition Dynamics) ---
    ax_lines = fig.add_subplot(gs[0, 0])
    
    # Select files with the most interesting transitions (high standard deviation)
    stds = np.std(all_profiles_matrix, axis=1)
    top_indices = np.argsort(stds)[::-1][:4]  # Top 4 files with most speed variations
    
    colors_lines = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
    time_axis = np.arange(1, 61)
    
    for i, idx in enumerate(top_indices):
        profile = all_profiles_matrix[idx]
        file_name = scooter_files[idx]
        lbl = os.path.basename(file_name)
        ax_lines.plot(time_axis, profile, color=colors_lines[i], linewidth=2.5, label=lbl, alpha=0.85)
        
    # Draw horizontal gear guides for fundamental speeds
    gears = [630.0, 700.0, 780.0, 856.0]
    gear_labels = ["Low Speed (Base - 630 Hz)", "Transition 1 (700 Hz)", "Transition 2 (780 Hz)", "High Speed (Turbo - 856 Hz)"]
    for val, g_lbl in zip(gears, gear_labels):
        ax_lines.axhline(val, color="#BDC3C7", linestyle=":", linewidth=0.8, alpha=0.6)
        ax_lines.annotate(g_lbl, (1.5, val + 5), fontsize=8.5, color="#7F8C8D", fontweight="bold")
        
    ax_lines.set_title("Selected Files: Dynamic Speed Transitions (Gear Shifts)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_lines.set_xlabel("Time (seconds)", fontsize=11)
    ax_lines.set_ylabel("Motor Hum Frequency (Hz)", fontsize=11)
    ax_lines.set_xlim(1, 60)
    ax_lines.set_ylim(580, 950)
    ax_lines.grid(True, linestyle="--", alpha=0.5)
    ax_lines.legend(loc="lower left", fontsize=9.5)
    
    # --- Panel 2: Global Speed Heatmap (All 42 Files Stacked) ---
    ax_heatmap = fig.add_subplot(gs[0, 1])
    img = ax_heatmap.imshow(all_profiles_matrix, cmap="plasma", aspect="auto", extent=[0.5, 60.5, len(scooter_files) - 0.5, -0.5], vmin=580, vmax=950)
    
    ax_heatmap.set_title("Global Dataset Speed Map (All 42 Scooter Files Stacked)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_heatmap.set_xlabel("Time in Recording (seconds)", fontsize=11)
    ax_heatmap.set_ylabel("Scooter File (Chronological Order)", fontsize=11)
    
    # Set y-ticks to show simplified file names
    y_ticks_indices = np.arange(len(scooter_files))
    ax_heatmap.set_yticks(y_ticks_indices)
    ax_heatmap.set_yticklabels(file_labels, fontsize=8)
    
    ax_heatmap.set_xlim(0.5, 60.5)
    
    # Add Colorbar
    cbar = fig.colorbar(img, ax=ax_heatmap, pad=0.03)
    cbar.set_label("Motor Hum Frequency (Hz) / Speed Level", fontsize=11)
    cbar.set_ticks(gears)
    cbar.set_ticklabels([f"{g_lbl}" for g_lbl in gear_labels])
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    
    print(f"\nSuccessfully generated and saved speed profile report:")
    print(f"  Local workspace path: {os.path.abspath(output_img)}")
    print(f"  Brain artifact path:  {os.path.abspath(artifact_path)}")

if __name__ == "__main__":
    main()
