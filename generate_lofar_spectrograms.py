import os
import glob
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.environ.get("RECORDINGS_DIR", "D:/RoyStudies/Recordings")
datasets = {
    "croatia_2507_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_1",
    "croatia_2507_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_2",
    "croatia_2407_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_1",
    "croatia_2407_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_2",
    "croatia_2307":   f"{BASE_DIR}/Croatia/Ocean Sonics/2307",
    "scooter":        f"{BASE_DIR}/DepartmentalCruise-2025-06-12/icListen/wav"
}

# Dark aesthetics matching premium design theme
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
TEXT_COL  = "#e0e0e0"
GRID_COL  = "#2d2d4e"

def setup_plot_style():
    plt.rcParams['figure.facecolor'] = DARK_BG
    plt.rcParams['axes.facecolor'] = PANEL_BG
    plt.rcParams['text.color'] = TEXT_COL
    plt.rcParams['axes.labelcolor'] = TEXT_COL
    plt.rcParams['xtick.color'] = TEXT_COL
    plt.rcParams['ytick.color'] = TEXT_COL
    plt.rcParams['axes.edgecolor'] = GRID_COL
    plt.rcParams['grid.color'] = GRID_COL

def generate_lofar(dataset_name, file_dir):
    print(f"\nProcessing LOFAR spectrogram for {dataset_name}...")
    wav_files = sorted(glob.glob(os.path.join(file_dir, "*.wav")))
    if not wav_files:
        print(f"No WAV files found in {file_dir}")
        return

    # Determine sample rate and configure frequencies
    info = sf.info(wav_files[0])
    sr = info.samplerate
    
    n_fft = 16 * 1024
    hop_length = n_fft // 2
    
    min_freq = 400 if dataset_name == "scooter" else 40
    max_freq = 2000
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_plot = freqs[freq_mask]
    
    # Process file by file to avoid running out of memory
    all_downsampled_cols = []
    total_samples = 0
    
    # We want ~3000 columns total for the full timeframe
    # Calculate downsample factor based on number of files
    # Croatia datasets: ~50-90 files. Each file has ~938 STFT columns.
    # Total columns ~50,000 to 90,000. 
    # Let's target ~30 columns per 60s file (downsample by 30)
    downsample_factor = 30
    
    for idx, fpath in enumerate(wav_files):
        if idx % 10 == 0:
            print(f"  [{idx}/{len(wav_files)}] Reading {os.path.basename(fpath)}")
            
        try:
            data, _ = sf.read(fpath, dtype='float32')
            if len(data.shape) > 1:
                y = np.mean(data, axis=1)
            else:
                y = data
            
            # STFT magnitude
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            mag = np.abs(stft[freq_mask, :])
            
            # Downsample the time axis (average blocks of columns)
            n_cols = mag.shape[1]
            n_new = n_cols // downsample_factor
            if n_new > 0:
                mag_down = mag[:, :n_new * downsample_factor].reshape(mag.shape[0], n_new, downsample_factor).mean(axis=2)
                all_downsampled_cols.append(mag_down)
            else:
                all_downsampled_cols.append(mag.mean(axis=1, keepdims=True))
                
            total_samples += len(y)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    if not all_downsampled_cols:
        print("No spectrogram columns collected.")
        return
        
    # Combine all downsampled chunks
    full_spec = np.hstack(all_downsampled_cols)
    total_duration = total_samples / sr
    
    # Convert to Decibels
    spec_db = librosa.amplitude_to_db(full_spec, ref=np.max)
    
    # LOFAR Background Normalization:
    # Subtract row-wise median to eliminate stationary frequency noise and bring out the narrowband tonals
    row_medians = np.median(spec_db, axis=1, keepdims=True)
    spec_db_normalized = spec_db - row_medians
    
    # Setup plotting
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot using a beautiful inferno color map
    # Dynamic range limit: focus on positive enhancements above the median background
    im = ax.imshow(
        spec_db_normalized,
        aspect="auto",
        origin="lower",
        extent=[0, total_duration / 60.0, freqs_plot[0], freqs_plot[-1]],
        cmap="inferno",
        vmin=-3,
        vmax=18
    )
    
    ax.set_xlabel("Time (minutes)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold", labelpad=8)
    
    # Title formatting
    title_ds = dataset_name.replace("_", " ").upper()
    ax.set_title(f"LOFAR Spectrogram (Normalized) — {title_ds}\n(Timeframe: {total_duration/60.0:.1f} mins | Denoised via Row Median Subtraction)", 
                 fontsize=13, fontweight="bold", pad=12)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.15, color=GRID_COL)
    
    # Colorbar styling
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Signal strength above background (dB)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    
    output_filename = f"lofar_{dataset_name}.png"
    output_path = os.path.join("output", output_filename)
    fig.savefig(output_path, bbox_inches="tight", dpi=180, facecolor=DARK_BG)
    plt.close(fig)
    print(f"Saved LOFAR spectrogram to: {output_path}")

def main():
    if not os.path.exists("output"):
        os.makedirs("output")
        
    for name, path in datasets.items():
        if os.path.exists(path):
            generate_lofar(name, path)
        else:
            print(f"Path does not exist: {path}")

if __name__ == "__main__":
    main()
