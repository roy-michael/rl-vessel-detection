import os
import glob
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.environ.get("RECORDINGS_DIR", "C:/Users/Roy/Recordings")
datasets = {
    "croatia_2507_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_1",
    "croatia_2507_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2507_2",
    "croatia_2407_1": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_1",
    "croatia_2407_2": f"{BASE_DIR}/Croatia/Ocean Sonics/2407_2",
    "croatia_2307":   f"{BASE_DIR}/Croatia/Ocean Sonics/2307",
    "scooter":        f"{BASE_DIR}/scooter"
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
    max_freq = 4000
    n_mels = 256
    
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=min_freq, fmax=max_freq)
    
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
            
            # Mel spectrogram magnitude
            melspec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=min_freq,
                fmax=max_freq
            )
            # mag is linear amplitude
            mag = np.sqrt(melspec)
            
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
    
    # ── Timezone and Metadata Annotation Processing ──────────────────────
    # We parse the file timestamps to map absolute recording times.
    # Note: Filenames are in UTC/GMT or local IL time. The Croatia metadata log uses Croatia local time (CEST = UTC+2).
    # Let's inspect the first wav file's timestamp.
    first_fname = os.path.basename(wav_files[0])
    import re
    from datetime import datetime, timedelta
    match = re.search(r"(\d{8})_(\d{6})", first_fname)
    
    metadata_notes = []
    if match and "croatia" in dataset_name:
        date_str, time_str = match.groups()
        file_dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        
        # We need to determine if filenames match UTC or IL time, and offset them to CEST (Croatia Time).
        # Let's check the date of the files:
        # 2307 starts at 09:46:00. CEST metadata: "23/7 1248-1304 (freediving)" and "ferry".
        # 09:46:00 UTC corresponds to 11:46:00 CEST.
        # 2407_1 starts at 09:11:00 (which is 11:11:00 CEST). Metadata: "24/7 1111-1128 (air malfunction)".
        # 2407_2 starts at 16:21:00 (which is 18:21:00 CEST).
        # 2507_1 starts at 08:01:00 (which is 10:01:00 CEST).
        # 2507_2 starts at 09:41:00 (which is 11:41:00 CEST).
        # This confirms: Filenames are in UTC/GMT. Croatia local time (CEST) is UTC + 2 hours.
        # We will apply a +2 hours offset to convert the filename UTC times to Croatia local times.
        utc_offset_hours = 2
        
        # Metadata dictionary with Croatia CEST time ranges (day, start_hour, start_min, end_hour, end_min, annotation)
        croatia_events = [
            (22, 13, 13, 13, 46, "Air Malfunction"),
            (22, 18, 47, 19, 44, "300m Route"),
            (23, 12, 48, 13, 4,  "Freediving (Clean/Ferry)"),
            (24, 11, 11, 11, 28, "Air Malfunction"),
            (24, 12, 12, 12, 58, "600m Route"),
            (24, 19, 22, 19, 57, "Snake Route"),
            (25, 11, 22, 12, 15, "1000m Route"),
            (25, 11, 46, 12, 7,  "Boat Noise")
        ]
        
        # Scan through the timeline and find overlaps
        # Each column's absolute time is: file_dt + timedelta(seconds=col_index * hop_length / sr) + timedelta(hours=2)
        # For simplicity, we compute time markers in minutes relative to start of the spectrogram
        start_cest = file_dt + timedelta(hours=utc_offset_hours)
        
        for ev_day, ev_sh, ev_sm, ev_eh, ev_em, label in croatia_events:
            if ev_day == start_cest.day:
                ev_start_dt = datetime(start_cest.year, start_cest.month, ev_day, ev_sh, ev_sm, 0)
                ev_end_dt = datetime(start_cest.year, start_cest.month, ev_day, ev_eh, ev_em, 0)
                
                # Convert to minutes relative to start_cest
                rel_start_min = (ev_start_dt - start_cest).total_seconds() / 60.0
                rel_end_min = (ev_end_dt - start_cest).total_seconds() / 60.0
                
                # If the event overlaps with our timeline (0 to total_duration / 60.0)
                total_duration_min = total_duration / 60.0
                if rel_end_min >= 0 and rel_start_min <= total_duration_min:
                    metadata_notes.append({
                        "start": max(0.0, rel_start_min),
                        "end": min(total_duration_min, rel_end_min),
                        "label": label,
                        "cest_str": f"{ev_sh:02d}:{ev_sm:02d}-{ev_eh:02d}:{ev_em:02d}"
                    })

    # Setup plotting
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(15, 7.5))
    
    # Plot using a beautiful inferno color map
    im = ax.imshow(
        spec_db_normalized,
        aspect="auto",
        origin="lower",
        extent=[0, total_duration / 60.0, 0, n_mels - 1],
        cmap="inferno",
        vmin=-3,
        vmax=18
    )
    
    # Overlay metadata events on the spectrogram
    for note in metadata_notes:
        ax.axvspan(note["start"], note["end"], color="cyan", alpha=0.12, edgecolor="cyan", linestyle="--", linewidth=1.0)
        # Draw a vertical marker line at start and end
        ax.axvline(note["start"], color="cyan", alpha=0.4, linestyle=":", linewidth=1.2)
        ax.axvline(note["end"], color="cyan", alpha=0.4, linestyle=":", linewidth=1.2)
        
        # Add a text label above the spectrogram panel or near the top
        mid_point = (note["start"] + note["end"]) / 2.0
        ax.text(mid_point, n_mels * 0.88, f"{note['label']}\n({note['cest_str']} Local)",
                color="cyan", fontsize=8.5, fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#16213e", ec="cyan", alpha=0.75))

    # Set custom Mel-spaced frequency ticks on y-axis
    freq_ticks = [100, 250, 500, 1000, 1500, 2000, 3000, 4000]
    freq_ticks = [f for f in freq_ticks if min_freq <= f <= max_freq]
    tick_positions = []
    for f in freq_ticks:
        idx = np.abs(mel_freqs - f).argmin()
        tick_positions.append(idx)
        
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{f}" for f in freq_ticks])
    
    ax.set_xlabel("Time (minutes relative to start)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Frequency (Hz) [Mel Scale]", fontsize=11, fontweight="bold", labelpad=8)
    
    # Title formatting
    title_ds = dataset_name.replace("_", " ").upper()
    
    # Format the subtitle to display absolute start time in CEST
    start_time_str = ""
    if match and "croatia" in dataset_name:
        start_time_str = f" | Start Time (CEST): {start_cest.strftime('%Y-%m-%d %H:%M:%S')}"
        
    ax.set_title(f"LOFAR Spectrogram (Normalized Mel Scale) — {title_ds}\n(Timeframe: {total_duration/60.0:.1f} mins | Denoised via Row Median Subtraction{start_time_str})", 
                 fontsize=13, fontweight="bold", pad=12)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.15, color=GRID_COL)
    
    # Colorbar styling
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Signal strength above background (dB)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    
    out_dir = os.path.join("output", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    output_filename = f"lofar_{dataset_name}.png"
    output_path = os.path.join(out_dir, output_filename)
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
