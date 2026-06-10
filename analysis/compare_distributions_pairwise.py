import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.signal import butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_signature_aware_segment(filepath, sr=16000, seg_len_sec=1.0):
    """
    Loads audio, high-pass filters it at 150 Hz, and scans 1-second segments.
    It selects the segment that maximizes the power spectrum peak in the 500 - 950 Hz range
    (representing motor hums).
    Returns the Z-score normalized raw samples of that segment.
    """
    # Load audio downsampled to 16 kHz
    y, sr_actual = librosa.load(filepath, sr=sr, mono=True)
    if len(y) < sr:
        raise ValueError(f"File too short: {len(y)} samples")
        
    win_len = int(seg_len_sec * sr)
    n_wins = len(y) // win_len
    
    # Apply high-pass filter at 150 Hz to remove low-frequency slope for signature detection
    y_hp = butter_highpass_filter(y, 150.0, sr, order=5)
    
    best_win_idx = 0
    max_spectral_val = -1.0
    
    # Store peak value for each window
    peak_vals = []
    rms_vals = []
    
    for i in range(n_wins):
        start_idx = i * win_len
        end_idx = start_idx + win_len
        
        # Calculate raw RMS
        window_raw = y[start_idx:end_idx]
        rms = np.sqrt(np.mean(window_raw ** 2))
        rms_vals.append(rms)
        
        # Calculate spectrum on high-pass filtered window
        window_hp = y_hp[start_idx:end_idx]
        
        std_val = np.std(window_hp)
        if std_val < 1e-10:
            peak_vals.append(0.0)
            continue
            
        windowed = window_hp * np.hanning(len(window_hp))
        spec = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
        mag = np.abs(spec)
        
        # Search band: 580 Hz to 950 Hz (excludes 539 Hz mooring line)
        idx_band = np.where((freqs >= 580.0) & (freqs <= 950.0))[0]
        if len(idx_band) > 0:
            hum_peak = np.max(mag[idx_band])
            peak_vals.append(hum_peak)
        else:
            peak_vals.append(0.0)
        
    # We choose the window that maximizes the hum signature peak
    best_win_idx = np.argmax(peak_vals)
    best_peak = peak_vals[best_win_idx]
    
    # If the strongest detected hum peak is extremely weak, fallback to the loudest RMS segment
    if best_peak < 1e-6:
        best_win_idx = np.argmax(rms_vals)
        
    start_idx = best_win_idx * win_len
    end_idx = start_idx + win_len
    best_raw_window = y[start_idx:end_idx]
    
    # Calculate original raw RMS
    raw_rms = np.sqrt(np.mean(best_raw_window ** 2))
    
    # Return Z-score normalized raw samples and original RMS
    std_raw = np.std(best_raw_window)
    if std_raw < 1e-10:
        std_raw = 1e-10
    norm_raw = (best_raw_window - np.mean(best_raw_window)) / std_raw
    return norm_raw, raw_rms

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40, fill='#'):
    """
    Call in a loop to create terminal progress bar.
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r', flush=True)
    if iteration == total:
        print()

def main():
    scooter_dir = r"C:\Users\Roy\Recordings\scooter"
    croatia_base_dir = r"C:\Users\Roy\Recordings\Croatia\Ocean Sonics"
    output_img = "amplitude_distribution_comparison.png"
    artifact_dir = "output"
    artifact_path = os.path.join(artifact_dir, output_img)
    
    # 1. Scan files recursively
    scooter_files = sorted(glob.glob(os.path.join(scooter_dir, "*.wav")))
    croatia_files = sorted(glob.glob(os.path.join(croatia_base_dir, "**", "*.wav"), recursive=True))
    
    print(f"Found {len(scooter_files)} Scooter files and {len(croatia_files)} total Croatia files across all subdirectories.")
    
    all_signals = []
    all_rms = []
    labels = []
    subdirs = []
    file_names = []
    
    total_files_to_load = len(scooter_files) + len(croatia_files)
    loaded_count = 0
    
    # Initialize progress bar for file loading
    print("\nLoading and analyzing all audio files...")
    print_progress_bar(loaded_count, total_files_to_load, prefix='Loading Files:', suffix='Complete', length=50)
    
    # Load Scooter files
    for f in scooter_files:
        try:
            sig, rms_val = get_signature_aware_segment(f)
            all_signals.append(sig)
            all_rms.append(rms_val)
            labels.append("Scooter")
            subdirs.append("Scooter")
            file_names.append(os.path.basename(f))
        except Exception as e:
            pass
        loaded_count += 1
        print_progress_bar(loaded_count, total_files_to_load, prefix='Loading Files:', suffix=f'({loaded_count}/{total_files_to_load} processed)', length=50)
            
    n_scooter_loaded = len(all_signals)
    
    # Group Croatia files by subdirectories
    croatia_by_subdir = {}
    for f in croatia_files:
        parent_name = os.path.basename(os.path.dirname(f))
        if parent_name not in croatia_by_subdir:
            croatia_by_subdir[parent_name] = []
        croatia_by_subdir[parent_name].append(f)
        
    # Load Croatia files sorted by subdirectory
    subdir_boundaries = []
    current_idx = n_scooter_loaded
    
    for folder in sorted(croatia_by_subdir.keys()):
        files_in_folder = sorted(croatia_by_subdir[folder])
        folder_loaded_count = 0
        
        for f in files_in_folder:
            try:
                sig, rms_val = get_signature_aware_segment(f)
                all_signals.append(sig)
                all_rms.append(rms_val)
                labels.append("Croatia")
                subdirs.append(folder)
                file_names.append(os.path.basename(f))
                folder_loaded_count += 1
            except Exception as e:
                pass
            loaded_count += 1
            print_progress_bar(loaded_count, total_files_to_load, prefix='Loading Files:', suffix=f'({loaded_count}/{total_files_to_load} processed)', length=50)
                
        if folder_loaded_count > 0:
            current_idx += folder_loaded_count
            subdir_boundaries.append((folder, current_idx))
            
    n_total = len(all_signals)
    n_croatia_loaded = n_total - n_scooter_loaded
    print(f"Successfully loaded {n_scooter_loaded} Scooter files and {n_croatia_loaded} Croatia files. Total: {n_total}")
    
    if n_total == 0:
        print("No files loaded successfully. Exiting.")
        return
        
    # 2. Compute pairwise Wasserstein distances
    print("\nComputing pairwise Wasserstein (Earth Mover's) distances...")
    dist_matrix = np.zeros((n_total, n_total))
    
    print_progress_bar(0, n_total, prefix='Calculating Distances:', suffix='Complete', length=50)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            dist = wasserstein_distance(all_signals[i][::16], all_signals[j][::16])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
        print_progress_bar(i + 1, n_total, prefix='Calculating Distances:', suffix=f'({i+1}/{n_total} completed)', length=50)
            
    print("Distance matrix computed.")
    
    # 3. Categorize distances for boxplots
    scooter_scooter_dists = []
    croatia_croatia_dists = []
    scooter_croatia_dists = []
    
    for i in range(n_total):
        for j in range(i + 1, n_total):
            dist_val = dist_matrix[i, j]
            label_i = labels[i]
            label_j = labels[j]
            
            if label_i == "Scooter" and label_j == "Scooter":
                scooter_scooter_dists.append(dist_val)
            elif label_i == "Croatia" and label_j == "Croatia":
                croatia_croatia_dists.append(dist_val)
            else:
                scooter_croatia_dists.append(dist_val)
                
    # 4. Compute average histograms for overlay plot
    print("\nComputing average histograms...")
    bin_range = (-4.0, 4.0)
    n_bins = 60
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    scooter_hists = []
    croatia_hists = []
    
    for i in range(n_total):
        hist, _ = np.histogram(all_signals[i], bins=bin_edges, density=True)
        if labels[i] == "Scooter":
            scooter_hists.append(hist)
        else:
            croatia_hists.append(hist)
            
    avg_scooter_hist = np.mean(scooter_hists, axis=0)
    avg_croatia_hist = np.mean(croatia_hists, axis=0)
    
    # 5. Plotting Setup (Modern Aesthetics)
    print("Generating visualization report...")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"
    plt.rcParams["xtick.color"] = "#2C3E50"
    plt.rcParams["ytick.color"] = "#2C3E50"
    
    fig = plt.figure(figsize=(18, 14), facecolor="#F8F9FA")
    fig.suptitle("Recursive EMD Amplitude & Level Comparison: Scooter vs. Croatia\nBased on Z-Score Normalized Waveform Shapes & Original Segment RMS Levels", 
                 fontsize=18, fontweight="bold", y=0.98, color="#1A252C")
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # --- Panel 1: Pairwise Distance Heatmap ---
    ax_heatmap = fig.add_subplot(gs[0, 0])
    img = ax_heatmap.imshow(dist_matrix, cmap="viridis", aspect="equal")
    ax_heatmap.set_title(f"Pairwise Wasserstein (EMD) Distance Matrix (Total Files: {n_total})", fontsize=13, fontweight="bold", color="#1F3A52")
    
    # Draw main boundary line separating Scooter and Croatia
    ax_heatmap.axvline(n_scooter_loaded - 0.5, color="#E74C3C", linestyle="-", linewidth=3.0, alpha=0.9, label="Scooter/Croatia Boundary")
    ax_heatmap.axhline(n_scooter_loaded - 0.5, color="#E74C3C", linestyle="-", linewidth=3.0, alpha=0.9)
    
    # Draw boundary lines for Croatia subdirectories
    for folder, idx in subdir_boundaries[:-1]:
        ax_heatmap.axvline(idx - 0.5, color="white", linestyle=":", linewidth=1.2, alpha=0.6)
        ax_heatmap.axhline(idx - 0.5, color="white", linestyle=":", linewidth=1.2, alpha=0.6)
        
    # Axis ticks and labels
    tick_locs = [n_scooter_loaded / 2]
    tick_labels = ["Scooter"]
    prev_boundary = n_scooter_loaded
    
    for folder, idx in subdir_boundaries:
        loc = prev_boundary + (idx - prev_boundary) / 2
        tick_locs.append(loc)
        tick_labels.append(folder)
        prev_boundary = idx
        
    ax_heatmap.set_xticks(tick_locs)
    ax_heatmap.set_xticklabels(tick_labels, fontsize=9, fontweight="bold", rotation=45, ha="right")
    ax_heatmap.set_yticks(tick_locs)
    ax_heatmap.set_yticklabels(tick_labels, fontsize=9, fontweight="bold")
    
    cbar = fig.colorbar(img, ax=ax_heatmap, pad=0.04, shrink=0.8)
    cbar.set_label("Wasserstein Distance (EMD)", fontsize=10)
    ax_heatmap.legend(loc="upper right", fontsize=8)
    
    # --- Panel 2: Distance Boxplots ---
    ax_box = fig.add_subplot(gs[0, 1])
    box_data = [scooter_scooter_dists, croatia_croatia_dists, scooter_croatia_dists]
    bp = ax_box.boxplot(box_data, tick_labels=["Scooter-to-Scooter\n(Intra-Scooter)", "Croatia-to-Croatia\n(Intra-Croatia)", "Scooter-to-Croatia\n(Inter-Dataset)"], 
                        patch_artist=True, widths=0.5)
    
    # Color boxplots
    colors = ["#DD8452", "#4C72B0", "#9B59B6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    ax_box.set_title("Wasserstein Distance Distribution Comparison", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_box.set_ylabel("Wasserstein Distance (EMD)", fontsize=11)
    ax_box.grid(True, linestyle="--", alpha=0.5)
    
    # --- Panel 3: Overlay Histograms ---
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.fill_between(bin_centers, avg_scooter_hist, color="#DD8452", alpha=0.4, label="Average Scooter Segment", step="mid")
    ax_hist.plot(bin_centers, avg_scooter_hist, color="#DD8452", linewidth=2, drawstyle="steps-mid")
    
    ax_hist.fill_between(bin_centers, avg_croatia_hist, color="#4C72B0", alpha=0.4, label="Average Croatia (All Folders)", step="mid")
    ax_hist.plot(bin_centers, avg_croatia_hist, color="#4C72B0", linewidth=2, drawstyle="steps-mid")
    
    # Also plot normal Gaussian for comparison
    gaussian = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * bin_centers ** 2)
    ax_hist.plot(bin_centers, gaussian, color="#7F8C8D", linestyle=":", linewidth=2, label="Perfect Gaussian (Normal)")
    
    ax_hist.set_title("Average Waveform Amplitude Probability Density (PDF)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_hist.set_xlabel("Amplitude (Z-Score Standard Deviations)", fontsize=11)
    ax_hist.set_ylabel("Probability Density", fontsize=11)
    ax_hist.grid(True, linestyle="--", alpha=0.5)
    ax_hist.set_xlim(bin_range[0], bin_range[1])
    ax_hist.legend(loc="upper right")
    
    # --- Panel 4: Used Signal RMS Amplitudes ---
    ax_amp = fig.add_subplot(gs[1, 1])
    x_indices = np.arange(n_total)
    rms_db = 20 * np.log10(np.array(all_rms) + 1e-10)
    
    # Find unique subdirs to group in legend
    unique_subdirs = []
    for sd in subdirs:
        if sd not in unique_subdirs:
            unique_subdirs.append(sd)
            
    color_map = {
        "Scooter": "#DD8452",
        "2307": "#4C72B0",
        "2407_1": "#55A868",
        "2407_2": "#C44E52",
        "2507_1": "#8172B3",
        "2507_2": "#937860"
    }
    fallback_colors = ["#1ABC9C", "#2ECC71", "#3498DB", "#9B59B6", "#F1C40F", "#E67E22"]
    
    for i, sd in enumerate(unique_subdirs):
        idx_mask = [j for j in range(n_total) if subdirs[j] == sd]
        color = color_map.get(sd, fallback_colors[i % len(fallback_colors)])
        ax_amp.scatter(x_indices[idx_mask], rms_db[idx_mask], color=color, alpha=0.7, label=sd, edgecolors="none", s=30)
        
    # Draw vertical boundary line separating Scooter and Croatia
    ax_amp.axvline(n_scooter_loaded - 0.5, color="#E74C3C", linestyle="-", linewidth=2.0, alpha=0.8, label="Scooter/Croatia Boundary")
    
    # Draw boundary lines for Croatia subdirectories
    for folder, idx in subdir_boundaries[:-1]:
        ax_amp.axvline(idx - 0.5, color="#7F8C8D", linestyle=":", linewidth=1.0, alpha=0.5)
        
    ax_amp.set_title("Selected Segment RMS Amplitude Level (dBFS)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_amp.set_xlabel("File Index (Sorted)", fontsize=11)
    ax_amp.set_ylabel("RMS Amplitude (dBFS)", fontsize=11)
    ax_amp.grid(True, linestyle="--", alpha=0.5)
    ax_amp.legend(loc="lower left", ncol=2, fontsize=8)
    
    mean_ss = np.mean(scooter_scooter_dists)
    mean_cc = np.mean(croatia_croatia_dists)
    mean_sc = np.mean(scooter_croatia_dists)
    print(f"\nMean Wasserstein Distances:")
    print(f"  Intra-Scooter: {mean_ss:.4f}")
    print(f"  Intra-Croatia: {mean_cc:.4f}")
    print(f"  Inter-Dataset: {mean_sc:.4f}")
    
    # Save the beautiful plot
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    
    print(f"\nSuccessfully generated and saved comparison diagram:")
    print(f"  Local workspace path: {os.path.abspath(output_img)}")
    print(f"  Brain artifact path:  {os.path.abspath(artifact_path)}")

if __name__ == "__main__":
    main()
