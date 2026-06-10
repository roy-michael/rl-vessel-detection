import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
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

def get_detailed_segment(filepath, sr=16000, seg_len_sec=1.0):
    """
    Loads audio, high-pass filters it at 150 Hz, and scans 1-second segments.
    Selects the segment that maximizes the cepstral peak in the 0.5 - 3.5 ms range.
    Returns:
        norm_raw: Z-score normalized raw samples of that segment
        raw_samples: original unnormalized raw samples of that segment
        ceps: cepstrum of that segment
        quefrency: quefrency axis in ms
        best_win_idx: index of the chosen window
    """
    y, sr_actual = librosa.load(filepath, sr=sr, mono=True)
    if len(y) < sr:
        raise ValueError(f"File too short: {len(y)} samples")
        
    win_len = int(seg_len_sec * sr)
    n_wins = len(y) // win_len
    
    # Apply high-pass filter at 150 Hz
    y_hp = butter_highpass_filter(y, 150.0, sr, order=5)
    
    min_quef_idx = int(0.5 / 1000.0 * sr)  # 8 samples (2.0 kHz hum)
    max_quef_idx = int(3.5 / 1000.0 * sr)  # 56 samples (285.7 Hz hum)
    
    peak_vals = []
    rms_vals = []
    
    for i in range(n_wins):
        start_idx = i * win_len
        end_idx = start_idx + win_len
        
        # Calculate raw RMS
        window_raw = y[start_idx:end_idx]
        rms = np.sqrt(np.mean(window_raw ** 2))
        rms_vals.append(rms)
        
        # Calculate cepstrum on high-pass filtered window
        window_hp = y_hp[start_idx:end_idx]
        std_val = np.std(window_hp)
        if std_val < 1e-10:
            std_val = 1e-10
        window_hp_norm = (window_hp - np.mean(window_hp)) / std_val
        
        windowed = window_hp_norm * np.hanning(len(window_hp_norm))
        spec = np.fft.fft(windowed)
        log_spec = np.log(np.abs(spec) + 1e-10)
        ceps_win = np.real(np.fft.ifft(log_spec))
        
        # Max value in the expanded hum region (0.5 ms to 3.5 ms)
        hum_peak = np.max(ceps_win[min_quef_idx:max_quef_idx])
        peak_vals.append(hum_peak)
        
    best_win_idx = np.argmax(peak_vals)
    best_peak = peak_vals[best_win_idx]
    
    if best_peak < 0.001:
        best_win_idx = np.argmax(rms_vals)
        
    start_idx = best_win_idx * win_len
    end_idx = start_idx + win_len
    
    best_raw_window = y[start_idx:end_idx]
    
    # Calculate cepstrum of the best window for output
    window_hp = y_hp[start_idx:end_idx]
    std_val = np.std(window_hp)
    if std_val < 1e-10:
        std_val = 1e-10
    window_hp_norm = (window_hp - np.mean(window_hp)) / std_val
    windowed = window_hp_norm * np.hanning(len(window_hp_norm))
    spec = np.fft.fft(windowed)
    log_spec = np.log(np.abs(spec) + 1e-10)
    ceps = np.real(np.fft.ifft(log_spec))
    quefrency = np.arange(len(ceps)) / sr * 1000.0  # in ms
    
    # Z-score normalize
    std_raw = np.std(best_raw_window)
    if std_raw < 1e-10:
        std_raw = 1e-10
    norm_raw = (best_raw_window - np.mean(best_raw_window)) / std_raw
    
    return norm_raw, best_raw_window, ceps, quefrency, best_win_idx

def main():
    # File paths
    scooter_path = r"C:\Users\Roy\Recordings\scooter\RBW6922_20250612_060400.wav"
    croatia_path = r"C:\Users\Roy\Recordings\Croatia\Ocean Sonics\2507_1\RBW6737_20250725_081300.wav"
    
    output_img = "signal_similarity_comparison.png"
    artifact_dir = "output"
    artifact_path = os.path.join(artifact_dir, output_img)
    
    print("Loading and analyzing representative Scooter segment...")
    sc_norm, sc_raw, sc_ceps, sc_quef, sc_win = get_detailed_segment(scooter_path)
    
    print("Loading and analyzing representative Croatia segment...")
    cr_norm, cr_raw, cr_ceps, cr_quef, cr_win = get_detailed_segment(croatia_path)
    
    # Modern clean styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"
    plt.rcParams["xtick.color"] = "#2C3E50"
    plt.rcParams["ytick.color"] = "#2C3E50"
    
    fig = plt.figure(figsize=(18, 15), facecolor="#F8F9FA")
    fig.suptitle("Direct Signal Comparison: Scooter vs. Croatia\nIdentifying Physical and Statistical Waveform Similarities", 
                 fontsize=18, fontweight="bold", y=0.98, color="#1A252C")
    
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
    
    # Time axes
    fs = 16000
    time_1s = np.linspace(0, 1.0, fs)
    zoom_len = int(0.05 * fs)  # 50 ms
    time_50ms = np.linspace(0, 50.0, zoom_len)
    
    # --- Row 1, Left: Scooter Waveform ---
    ax_sc_wave = fig.add_subplot(gs[0, 0])
    ax_sc_wave.plot(time_1s, sc_norm, color="#DD8452", linewidth=0.5, alpha=0.8)
    ax_sc_wave.set_title(f"Scooter Representative Segment Waveform (1 Second, Z-Normalized)\nFile: {os.path.basename(scooter_path)} [Window {sc_win}]", 
                         fontsize=12, fontweight="bold", color="#1F3A52")
    ax_sc_wave.set_xlabel("Time (seconds)", fontsize=10)
    ax_sc_wave.set_ylabel("Amplitude (Z-Score)", fontsize=10)
    ax_sc_wave.grid(True, linestyle="--", alpha=0.5)
    ax_sc_wave.set_xlim(0, 1.0)
    
    # --- Row 1, Right: Croatia Waveform ---
    ax_cr_wave = fig.add_subplot(gs[0, 1])
    ax_cr_wave.plot(time_1s, cr_norm, color="#4C72B0", linewidth=0.5, alpha=0.8)
    ax_cr_wave.set_title(f"Croatia Representative Segment Waveform (1 Second, Z-Normalized)\nFile: {os.path.basename(croatia_path)} [Window {cr_win}]", 
                         fontsize=12, fontweight="bold", color="#1F3A52")
    ax_cr_wave.set_xlabel("Time (seconds)", fontsize=10)
    ax_cr_wave.set_ylabel("Amplitude (Z-Score)", fontsize=10)
    ax_cr_wave.grid(True, linestyle="--", alpha=0.5)
    ax_cr_wave.set_xlim(0, 1.0)
    
    # --- Row 2, Left: Scooter Zoomed Waveform (50 ms) ---
    ax_sc_zoom = fig.add_subplot(gs[1, 0])
    ax_sc_zoom.plot(time_50ms, sc_norm[:zoom_len], color="#DD8452", linewidth=1.5)
    ax_sc_zoom.set_title("Scooter Waveform Zoom-in (50 ms) - Resolving Electric Motor & Propeller Pulses", 
                         fontsize=12, fontweight="bold", color="#1F3A52")
    ax_sc_zoom.set_xlabel("Time (milliseconds)", fontsize=10)
    ax_sc_zoom.set_ylabel("Amplitude (Z-Score)", fontsize=10)
    ax_sc_zoom.grid(True, linestyle="--", alpha=0.5)
    ax_sc_zoom.set_xlim(0, 50.0)
    
    # --- Row 2, Right: Croatia Zoomed Waveform (50 ms) ---
    ax_cr_zoom = fig.add_subplot(gs[1, 1])
    ax_cr_zoom.plot(time_50ms, cr_norm[:zoom_len], color="#4C72B0", linewidth=1.5)
    ax_cr_zoom.set_title("Croatia Waveform Zoom-in (50 ms) - Resolving Electric Motor & Propeller Pulses", 
                         fontsize=12, fontweight="bold", color="#1F3A52")
    ax_cr_zoom.set_xlabel("Time (milliseconds)", fontsize=10)
    ax_cr_zoom.set_ylabel("Amplitude (Z-Score)", fontsize=10)
    ax_cr_zoom.grid(True, linestyle="--", alpha=0.5)
    ax_cr_zoom.set_xlim(0, 50.0)
    
    # --- Row 3, Left: Overlaid PDF Amplitude Distribution ---
    ax_pdf = fig.add_subplot(gs[2, 0])
    bin_range = (-4.0, 4.0)
    n_bins = 60
    bin_edges = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    sc_hist, _ = np.histogram(sc_norm, bins=bin_edges, density=True)
    cr_hist, _ = np.histogram(cr_norm, bins=bin_edges, density=True)
    gaussian = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * bin_centers ** 2)
    
    ax_pdf.fill_between(bin_centers, sc_hist, color="#DD8452", alpha=0.3, label="Scooter Segment PDF", step="mid")
    ax_pdf.plot(bin_centers, sc_hist, color="#DD8452", linewidth=2.0, drawstyle="steps-mid")
    
    ax_pdf.fill_between(bin_centers, cr_hist, color="#4C72B0", alpha=0.3, label="Croatia Segment PDF", step="mid")
    ax_pdf.plot(bin_centers, cr_hist, color="#4C72B0", linewidth=2.0, drawstyle="steps-mid")
    
    ax_pdf.plot(bin_centers, gaussian, color="#7F8C8D", linestyle=":", linewidth=2, label="Perfect Gaussian (Reference)")
    
    ax_pdf.set_title("Overlaid Waveform Amplitude Probability Density Functions (PDF)", 
                     fontsize=12, fontweight="bold", color="#1F3A52")
    ax_pdf.set_xlabel("Amplitude (Z-Score / Standard Deviations)", fontsize=10)
    ax_pdf.set_ylabel("Probability Density", fontsize=10)
    ax_pdf.grid(True, linestyle="--", alpha=0.5)
    ax_pdf.set_xlim(bin_range[0], bin_range[1])
    ax_pdf.legend(loc="upper right")
    
    # --- Row 3, Right: Overlaid Real Cepstrums (0.5 to 3.5 ms region) ---
    ax_ceps = fig.add_subplot(gs[2, 1])
    
    # Get index range for 0.5 to 3.5 ms
    q_min_idx = int(0.5 / 1000.0 * fs)
    q_max_idx = int(3.5 / 1000.0 * fs)
    
    seg_quef_ms = sc_quef[q_min_idx:q_max_idx]
    
    ax_ceps.plot(seg_quef_ms, sc_ceps[q_min_idx:q_max_idx], color="#DD8452", linewidth=1.8, label="Scooter Segment Cepstrum")
    ax_ceps.plot(seg_quef_ms, cr_ceps[q_min_idx:q_max_idx], color="#4C72B0", linewidth=1.8, alpha=0.8, label="Croatia Segment Cepstrum")
    
    # Find peaks in the motor hum region to label them
    # For Scooter
    sc_subset = sc_ceps[q_min_idx:q_max_idx]
    sc_peak_local_idx = np.argmax(sc_subset)
    sc_peak_ms = seg_quef_ms[sc_peak_local_idx]
    sc_peak_val = sc_subset[sc_peak_local_idx]
    sc_peak_hz = 1000.0 / sc_peak_ms
    
    # For Croatia
    cr_subset = cr_ceps[q_min_idx:q_max_idx]
    cr_peak_local_idx = np.argmax(cr_subset)
    cr_peak_ms = seg_quef_ms[cr_peak_local_idx]
    cr_peak_val = cr_subset[cr_peak_local_idx]
    cr_peak_hz = 1000.0 / cr_peak_ms
    
    # Plot dots at peaks
    ax_ceps.plot(sc_peak_ms, sc_peak_val, "o", color="#C0392B", markersize=6)
    ax_ceps.annotate(f"Scooter Hum Peak:\n{sc_peak_ms:.2f} ms ({sc_peak_hz:.1f} Hz)",
                     (sc_peak_ms, sc_peak_val), xytext=(-10, 15), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="#C0392B"), fontsize=9, color="#C0392B", fontweight="bold")
                     
    ax_ceps.plot(cr_peak_ms, cr_peak_val, "o", color="#273746", markersize=6)
    ax_ceps.annotate(f"Croatia Hum Peak:\n{cr_peak_ms:.2f} ms ({cr_peak_hz:.1f} Hz)",
                     (cr_peak_ms, cr_peak_val), xytext=(10, -25), textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color="#273746"), fontsize=9, color="#273746", fontweight="bold")
    
    ax_ceps.set_title("Overlaid Real Cepstrums (0.5 - 3.5 ms Hum Signature Window)", 
                     fontsize=12, fontweight="bold", color="#1F3A52")
    ax_ceps.set_xlabel("Quefrency (milliseconds)", fontsize=10)
    ax_ceps.set_ylabel("Cepstral Coefficient Value", fontsize=10)
    ax_ceps.grid(True, linestyle="--", alpha=0.5)
    ax_ceps.set_xlim(0.5, 3.5)
    ax_ceps.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    
    print(f"Successfully generated comparison plot:")
    print(f"  Local path: {os.path.abspath(output_img)}")
    print(f"  Brain path:  {os.path.abspath(artifact_path)}")

if __name__ == "__main__":
    main()
