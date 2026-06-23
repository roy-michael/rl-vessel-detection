import os
import glob
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = os.environ.get("RECORDINGS_DIR", "D:/RoyStudies/Recordings")
DATASETS = {
    "croatia_2507_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_1_1k",
    "croatia_2507_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_2_joint",
    "croatia_2407_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_1_600m",
    "croatia_2407_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_2_snake",
    "croatia_2307":   f"{BASE_DIR}/Croatia/Ocean Sonics/2307_free",
    "scooter":        f"{BASE_DIR}/DepartmentalCruise-2025-06-12/icListen/wav"
}

def generate_for_dataset(name, path):
    print(f"\nProcessing full-session spectrograms for: {name} in {path}")
    wav_files = sorted(glob.glob(os.path.join(path, "*.wav")))
    if not wav_files:
        print(f"  No .wav files found for {name}!")
        return

    # 1. Setup sample rate and parameters from the first file
    try:
        info = sf.info(wav_files[0])
        sr = info.samplerate
    except Exception as e:
        print(f"  Error reading first file info: {e}")
        return

    n_fft = 16 * 1024
    hop_length = n_fft // 2
    
    # Target total columns for the entire session plot (increased by 50% for higher resolution)
    target_total_cols = 3000
    cols_per_file = max(5, target_total_cols // len(wav_files))

    lofar_cols = []
    mel_cols = []
    total_duration = 0.0

    print(f"  Processing {len(wav_files)} files (target: {cols_per_file} cols/file)...")
    for idx, fpath in enumerate(wav_files):
        try:
            # Read info first to skip corrupted files
            file_info = sf.info(fpath)
            # Read and average channels if stereo
            data, _ = sf.read(fpath, dtype='float32')
            if len(data.shape) > 1:
                y = np.mean(data, axis=1)
            else:
                y = data
            
            total_duration += len(y) / sr

            # --- LOFAR calculation (Normalized Mel Scale) ---
            melspec_lofar = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=256,
                fmin=40,
                fmax=2500
            )
            mag_lofar = np.sqrt(melspec_lofar)
            
            # --- Mel calculation (Raw Power) ---
            melspec_mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=256,
                fmin=40,
                fmax=2500
            )
            mag_mel = np.sqrt(melspec_mel)

            # Downsample helper: use mean along the time axis to match the reference image style
            def downsample_time(mag, target_cols):
                n_cols = mag.shape[1]
                ds_factor = max(1, n_cols // target_cols)
                n_new = n_cols // ds_factor
                if n_new > 0:
                    return mag[:, :n_new * ds_factor].reshape(mag.shape[0], n_new, ds_factor).mean(axis=2)
                return mag.mean(axis=1, keepdims=True)

            lofar_cols.append(downsample_time(mag_lofar, cols_per_file))
            mel_cols.append(downsample_time(mag_mel, cols_per_file))

        except Exception as e:
            # Silently skip bad/corrupted files
            continue

    if not lofar_cols:
        print("  Error: No valid spectrogram columns were computed.")
        return

    # Concatenate all downsampled files
    full_lofar = np.hstack(lofar_cols)
    full_mel = np.hstack(mel_cols)
    time_minutes = total_duration / 60.0

    out_dir = f"output/{name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Plot LOFAR Spectrogram (Normalized Mel Scale, 0 - 2.5 kHz) ---
    lofar_db = librosa.amplitude_to_db(full_lofar, ref=np.max)
    # Background subtraction to highlight tonals
    row_medians = np.median(lofar_db, axis=1, keepdims=True)
    lofar_db_norm = lofar_db - row_medians

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    # Color scaling matching the reference image (96th and 4th percentiles)
    vmax = np.percentile(lofar_db_norm, 96.0)
    vmin = np.percentile(lofar_db_norm, 4.0)

    img = ax.imshow(
        lofar_db_norm,
        aspect='auto',
        origin='lower',
        extent=[0, time_minutes, 0, 255],
        cmap='inferno',
        vmin=vmin,
        vmax=vmax
    )

    # Set custom Mel-spaced frequency ticks on y-axis
    mel_freqs_lofar = librosa.mel_frequencies(n_mels=256, fmin=40, fmax=2500)
    ticks_hz = [100, 250, 500, 1000, 1500, 2000, 2500]
    tick_pos = [np.abs(mel_freqs_lofar - hz).argmin() for hz in ticks_hz]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(hz) for hz in ticks_hz])
    ax.grid(True, which='both', color='#2d2d4e', linestyle='--', alpha=0.3)

    ax.set_title(f"LOFAR Spectrogram (Normalized Mel Scale) — {name.upper()}", color='white', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Time (minutes)", color='white', fontsize=10)
    ax.set_ylabel("Frequency (Hz) [Mel Scale]", color='white', fontsize=10)
    ax.tick_params(colors='white', which='both', labelsize=9)

    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.set_label("Signal strength above background (dB)", color='white')
    cbar.ax.tick_params(colors='white')

    lofar_path = f"{out_dir}/{name}_lofar.png"
    plt.savefig(lofar_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  Saved LOFAR spectrogram to: {lofar_path}")

    # --- 2. Plot Mel Spectrogram (0 - 2.5 kHz) ---
    mel_db = librosa.power_to_db(full_mel**2, ref=np.max)
    # Background subtraction to highlight tonals and black out ambient noise
    mel_medians = np.median(mel_db, axis=1, keepdims=True)
    mel_db_norm = mel_db - mel_medians

    # Color scaling matching the LOFAR spectrogram
    vmax_mel = max(15.0, np.percentile(mel_db_norm, 99.9))
    vmin_mel = -3.0

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
    ax.set_facecolor('#16213e')

    # Compute mel frequencies for scaling
    mel_freqs = librosa.mel_frequencies(n_mels=256, fmin=40, fmax=2500)

    # Plot as image with extent
    img_mel = ax.imshow(
        mel_db_norm,
        aspect='auto',
        origin='lower',
        extent=[0, time_minutes, 0, 255],
        cmap='inferno',
        vmin=vmin_mel,
        vmax=vmax_mel
    )

    # Set custom ticks on Mel axis
    ticks_hz = [100, 250, 500, 1000, 1500, 2000, 2500]
    tick_pos = [np.abs(mel_freqs - hz).argmin() for hz in ticks_hz]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(hz) for hz in ticks_hz])
    ax.grid(True, which='both', color='#2d2d4e', linestyle='--', alpha=0.3)

    ax.set_title(f"Mel-based Spectrogram (Normalized) — {name.upper()}", color='white', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Time (minutes)", color='white', fontsize=10)
    ax.set_ylabel("Frequency (Hz) [Mel Scale]", color='white', fontsize=10)
    ax.tick_params(colors='white', which='both', labelsize=9)

    cbar = fig.colorbar(img_mel, ax=ax, pad=0.02)
    cbar.set_label("Relative Amplitude above background (dB)", color='white')
    cbar.ax.tick_params(colors='white')

    mel_path = f"{out_dir}/{name}_melspec.png"
    plt.savefig(mel_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  Saved Mel-based spectrogram to: {mel_path}")

def main():
    for name, path in DATASETS.items():
        if os.path.exists(path):
            try:
                generate_for_dataset(name, path)
            except Exception as e:
                print(f"Error processing {name}: {e}")
        else:
            print(f"Path does not exist: {path}")

if __name__ == "__main__":
    main()
