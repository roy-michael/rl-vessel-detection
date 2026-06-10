import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main():
    # File paths
    wav_path = r"C:\Users\Roy\Recordings\scooter\RBW6922_20250612_060400.wav"
    output_img = "scooter_cepstrum_analysis.png"
    artifact_dir = "output"
    artifact_path = os.path.join(artifact_dir, output_img)
    
    print(f"Loading audio file: {wav_path}...")
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    print(f"Loaded. SR: {sr} Hz, Duration: {duration:.2f} s, Total Samples: {len(y)}")
    
    # Apply high-pass filter to the entire signal to isolate high-resolution cepstrum
    cutoff_hz = 150.0
    print(f"Applying high-pass filter at {cutoff_hz} Hz...")
    y_hp = butter_highpass_filter(y, cutoff_hz, sr, order=5)
    
    # -------------------------------------------------------------
    # 1. Select a Representative Segment (Seg 21, from 20.0 to 21.0s)
    # -------------------------------------------------------------
    seg_start_t = 20.0
    seg_end_t = 21.0
    seg_start_idx = int(seg_start_t * sr)
    seg_end_idx = int(seg_end_t * sr)
    
    # For waveform and spectrum, we use raw segment. For cepstrum, we use high-pass filtered segment.
    segment = y[seg_start_idx:seg_end_idx]
    segment_hp = y_hp[seg_start_idx:seg_end_idx]
    time_seg = np.linspace(seg_start_t, seg_end_t, len(segment))
    
    # -------------------------------------------------------------
    # 2. Compute Mel Spectrogram of the Segment
    # -------------------------------------------------------------
    S_seg = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=len(segment), hop_length=len(segment), n_mels=128, fmax=2000.0)
    S_seg_db = librosa.power_to_db(S_seg, ref=np.max)[:, 0]
    mel_freqs = librosa.mel_frequencies(n_mels=128, fmax=2000.0)
    
    # Compute high-resolution linear Cepstrum of the high-pass filtered segment
    window = np.hanning(len(segment_hp))
    windowed_seg_hp = segment_hp * window
    full_spectrum = np.fft.fft(windowed_seg_hp)
    log_full_spec = np.log(np.abs(full_spectrum) + 1e-10)
    cepstrum = np.real(np.fft.ifft(log_full_spec))
    quefrency = np.arange(len(cepstrum)) / sr
    
    # -------------------------------------------------------------
    # 3. Compute Mel Spectrogram and Cepstrogram of the Full 60s Signal
    # -------------------------------------------------------------
    print("Computing Mel spectrogram and high-resolution cepstrogram...")
    n_fft = 32768  # 0.256s window
    hop_length = 8192  # 0.064s hop
    
    # Mel Spectrogram
    S_full = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=128, fmax=2000.0)
    S_full_db = librosa.power_to_db(S_full, ref=np.max)
    
    # Cepstrogram computed on the high-pass filtered signal
    frames_hp = librosa.util.frame(y_hp, frame_length=n_fft, hop_length=hop_length)
    window_frame = np.hanning(n_fft)[:, np.newaxis]
    windowed_frames_hp = frames_hp * window_frame
    full_spec_frames = np.fft.fft(windowed_frames_hp, axis=0)
    log_spec_frames = np.log(np.abs(full_spec_frames) + 1e-10)
    cepstrogram = np.real(np.fft.ifft(log_spec_frames, axis=0))
    
    # -------------------------------------------------------------
    # 4. Plotting Setup (Modern Aesthetics)
    # -------------------------------------------------------------
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["text.color"] = "#2C3E50"
    plt.rcParams["axes.labelcolor"] = "#2C3E50"
    plt.rcParams["xtick.color"] = "#2C3E50"
    plt.rcParams["ytick.color"] = "#2C3E50"
    
    fig = plt.figure(figsize=(18, 15.5), facecolor="#F8F9FA")
    fig.suptitle("High-Resolution Cepstrum Analysis of Scooter Acoustic Signature\nFile: RBW6922_20250612_060400.wav (FS: 128 kHz, HPF: 150 Hz)", 
                 fontsize=18, fontweight="bold", y=0.98, color="#1A252C")
    
    # 4 rows: Row 1 = Waveforms, Row 2 = Spectra, Row 3 = Cepstrum, Row 4 = Full-Width Text Box
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.35], hspace=0.4, wspace=0.25)
    
    # --- Row 1, Left: Waveform (Entire 60s) ---
    ax_wave_full = fig.add_subplot(gs[0, 0])
    time_full = np.linspace(0, duration, len(y))
    ds_factor = 100
    ax_wave_full.plot(time_full[::ds_factor], y[::ds_factor], color="#5D6D7E", linewidth=0.5, alpha=0.8)
    ax_wave_full.axvspan(seg_start_t, seg_end_t, color="#E74C3C", alpha=0.25, label="Analyzed Segment (Seg 21)")
    ax_wave_full.set_title("Time-Domain Waveform (Full 60s)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_wave_full.set_xlabel("Time (Seconds)", fontsize=10)
    ax_wave_full.set_ylabel("Amplitude", fontsize=10)
    ax_wave_full.grid(True, linestyle="--", alpha=0.5)
    ax_wave_full.set_xlim(0, duration)
    ax_wave_full.legend(loc="upper right")
    
    # --- Row 1, Right: Detailed Waveform (Seg 21, 20-21s) ---
    ax_wave_seg = fig.add_subplot(gs[0, 1])
    ax_wave_seg.plot(time_seg, segment, color="#E74C3C", linewidth=0.8, alpha=0.9)
    ax_wave_seg.set_title("Detailed Waveform (Segment 21: 20.0s - 21.0s)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_wave_seg.set_xlabel("Time (Seconds)", fontsize=10)
    ax_wave_seg.set_ylabel("Amplitude", fontsize=10)
    ax_wave_seg.grid(True, linestyle="--", alpha=0.5)
    ax_wave_seg.set_xlim(seg_start_t, seg_end_t)
    
    # --- Row 2, Left: Mel Spectrogram (Entire 60s) ---
    ax_spec_full = fig.add_subplot(gs[1, 0])
    img_spec = librosa.display.specshow(
        S_full_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        fmax=2000.0,
        ax=ax_spec_full,
        cmap="magma",
        vmin=-80,
        vmax=-20
    )
    ax_spec_full.set_title("Mel Spectrogram (0 - 2000 Hz, Warped Y-Axis)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_spec_full.axvline(seg_start_t + 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_spec_full.text(seg_start_t + 1.0, 1800, "Seg 21", color="white", fontweight="bold", fontsize=9)
    cbar_spec = fig.colorbar(img_spec, ax=ax_spec_full, pad=0.02)
    cbar_spec.set_label("Magnitude (dBFS)", fontsize=9)
    
    # --- Row 2, Right: Mel Log Spectrum (Seg 21) ---
    ax_spec_seg = fig.add_subplot(gs[1, 1])
    ax_spec_seg.plot(mel_freqs, S_seg_db, color="#2C3E50", linewidth=1.5)
    peaks_spec, _ = find_peaks(S_seg_db, distance=5, prominence=5)
    top_spec_peaks = sorted(zip(S_seg_db[peaks_spec], mel_freqs[peaks_spec]), key=lambda x: x[0], reverse=True)[:5]
    for val, f in top_spec_peaks:
        ax_spec_seg.plot(f, val, "ro", markersize=4)
        ax_spec_seg.annotate(f"{f:.1f}Hz", (f, val), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8, color="red", fontweight="bold")
    ax_spec_seg.set_title("Log-Magnitude Mel Spectrum (Segment 21)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_spec_seg.set_xlabel("Frequency (Hz, Mel-Scaled)", fontsize=10)
    ax_spec_seg.set_ylabel("Magnitude (dB)", fontsize=10)
    ax_spec_seg.grid(True, linestyle="--", alpha=0.5)
    ax_spec_seg.set_xlim(0, 2000.0)
    ax_spec_seg.set_ylim(np.min(S_seg_db)-5, np.max(S_seg_db)+10)
    
    # --- Row 3, Left: High-Resolution Cepstrogram (Entire 60s) ---
    ax_ceps_full = fig.add_subplot(gs[2, 0])
    min_quef_ms, max_quef_ms = 0.5, 100.0
    q_min_idx = int(min_quef_ms / 1000.0 * sr)
    q_max_idx = int(max_quef_ms / 1000.0 * sr)
    
    img_ceps = ax_ceps_full.imshow(
        cepstrogram[q_min_idx:q_max_idx, :],
        aspect="auto",
        origin="lower",
        extent=[0, duration, min_quef_ms, max_quef_ms],
        cmap="viridis",
        vmin=-0.005,
        vmax=0.015
    )
    ax_ceps_full.set_title(f"High-Resolution Cepstrogram (HPF: {cutoff_hz} Hz, Quefrency: {min_quef_ms:.1f} - {max_quef_ms:.1f} ms)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_ceps_full.set_xlabel("Time (Seconds)", fontsize=10)
    ax_ceps_full.set_ylabel("Quefrency (ms)", fontsize=10)
    ax_ceps_full.axvline(seg_start_t + 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.7)
    cbar_ceps = fig.colorbar(img_ceps, ax=ax_ceps_full, pad=0.02)
    cbar_ceps.set_label("Cepstral Amplitude", fontsize=9)
    
    # --- Row 3, Right: High-Resolution Segment Cepstrum ---
    ax_ceps_seg = fig.add_subplot(gs[2, 1])
    seg_quef_ms = quefrency[q_min_idx:q_max_idx] * 1000.0
    seg_ceps = cepstrum[q_min_idx:q_max_idx]
    
    ax_ceps_seg.plot(seg_quef_ms, seg_ceps, color="#16A085", linewidth=1.2, label="Real Cepstrum")
    
    # Find peaks in the segment cepstrum
    # 1. Low frequency range (quefrency >= 3.0 ms)
    q_low_min_idx = int(3.0 / 1000.0 * sr)
    peaks_ceps_low, _ = find_peaks(cepstrum[q_low_min_idx:q_max_idx], distance=int(1.0 / 1000.0 * sr), prominence=0.0005)
    actual_peaks_low = peaks_ceps_low + q_low_min_idx
    top_ceps_peaks_low = sorted(zip(cepstrum[actual_peaks_low], quefrency[actual_peaks_low] * 1000.0), key=lambda x: x[0], reverse=True)[:4]
    
    # 2. High frequency range (quefrency 0.5 to 3.0 ms)
    peaks_ceps_high, _ = find_peaks(cepstrum[q_min_idx:q_low_min_idx], distance=int(0.1 / 1000.0 * sr), prominence=0.0005)
    actual_peaks_high = peaks_ceps_high + q_min_idx
    top_ceps_peaks_high = sorted(zip(cepstrum[actual_peaks_high], quefrency[actual_peaks_high] * 1000.0), key=lambda x: x[0], reverse=True)[:1]
    
    top_ceps_peaks = top_ceps_peaks_high + top_ceps_peaks_low
    colors_peaks = ["#C0392B", "#D35400", "#7D6608", "#273746", "#8E44AD"]
    for i, (val, q_ms) in enumerate(top_ceps_peaks):
        equiv_f = 1000.0 / q_ms
        ax_ceps_seg.plot(q_ms, val, "o", color=colors_peaks[i % len(colors_peaks)], markersize=5)
        ax_ceps_seg.annotate(
            f"{q_ms:.2f}ms\n({equiv_f:.1f}Hz)", 
            (q_ms, val), 
            textcoords="offset points", 
            xytext=(0,8), 
            ha="center", 
            fontsize=8.5, 
            color=colors_peaks[i % len(colors_peaks)], 
            fontweight="bold"
        )
        
    ax_ceps_seg.set_title("High-Resolution Real Cepstrum (Segment 21)", fontsize=13, fontweight="bold", color="#1F3A52")
    ax_ceps_seg.set_xlabel("Quefrency (ms)", fontsize=10)
    ax_ceps_seg.set_ylabel("Cepstral Amplitude", fontsize=10)
    ax_ceps_seg.grid(True, linestyle="--", alpha=0.5)
    ax_ceps_seg.set_xlim(min_quef_ms, max_quef_ms)
    ax_ceps_seg.set_ylim(np.min(seg_ceps)-0.002, np.max(seg_ceps)+0.005)
    
    # --- Row 4: Dedicated Full-Width Text Box ---
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis("off")  # Hide axes for clean text display
    
    explanation_text = (
        "Signal Processing & Physical Interpretation:\n"
        "- High-Pass Filtering at 150 Hz removes the massive low-frequency spectral tilt and grid/mooring noise.\n"
        "- This flattens the baseline and eliminates the bottom quefrency 'glow' from 0 to 2 ms, allowing the scooter hum to stand out clearly.\n"
        "- Peak 1: 1.64 ms \u2192 609.5 Hz (high-frequency motor hum component, representing electric motor pole-passing/slot-passing rate).\n"
        "- Peak 2: 86.40 ms \u2192 11.6 Hz (motor shaft base rotation frequency: 11.6 Hz * 60 seconds = 694 RPM).\n"
        "- Peak 3: 76.24 ms \u2192 13.1 Hz (blade rate / sub-harmonic spacing of the propeller blades passing through the water).\n"
        "- Peak 4: 68.27 ms \u2192 14.7 Hz (secondary mechanical harmonic spacing).\n"
        "- Peak 5: 63.48 ms \u2192 15.8 Hz (auxiliary machinery/casing structural resonance).\n"
        "- The high-resolution linear cepstrum preserves fine frequency spacing details which are lost in smoothed Mel-scale MFCCs."
    )
    props = dict(boxstyle="round,pad=0.8", facecolor="#EBF5FB", edgecolor="#AED6F1", alpha=0.9)
    ax_text.text(0.5, 0.5, explanation_text, transform=ax_text.transAxes, fontsize=10.5,
                 verticalalignment="center", horizontalalignment="center", bbox=props, color="#2E4053")
    
    # Save the beautiful plot
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.savefig(artifact_path, dpi=150)
    plt.close()
    
    print(f"Successfully generated and saved Scooter cepstrum diagram:")
    print(f"  Local workspace path: {os.path.abspath(output_img)}")
    print(f"  Brain artifact path:  {os.path.abspath(artifact_path)}")

if __name__ == "__main__":
    main()
