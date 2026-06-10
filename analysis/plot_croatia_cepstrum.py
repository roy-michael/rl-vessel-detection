"""
Two-column cepstrum comparison:
  Left  column: RBW6737_20250725_081200.wav  (vessel onset / approaching)
  Right column: RBW6737_20250725_081300.wav  (vessel peak pass)
Each column shows: waveform | Mel spectrogram | real cepstrum of representative segment.
"""
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, lfilter

BASE = r"C:\Users\Roy\Recordings\Croatia\Ocean Sonics\2507_1"
FILES = {
    "onset":     (os.path.join(BASE, "RBW6737_20250725_081200.wav"), 22.0, "#E67E22"),
    "peak_pass": (os.path.join(BASE, "RBW6737_20250725_081300.wav"), 34.0, "#4C72B0"),
}
OUTPUT_IMG = os.path.join("output", "croatia_cepstrum_comparison.png")


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high", analog=False)
    return lfilter(b, a, data)


def process_file(wav_path, seg_start_t, color):
    """Load, filter, and compute all quantities needed for the three panels."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr
    cutoff_hz = 150.0
    y_hp = butter_highpass_filter(y, cutoff_hz, sr)

    seg_end_t = seg_start_t + 1.0
    i0, i1 = int(seg_start_t * sr), int(seg_end_t * sr)
    segment    = y[i0:i1]
    segment_hp = y_hp[i0:i1]
    time_seg   = np.linspace(seg_start_t, seg_end_t, len(segment))

    # Mel spectrogram (full 60 s)
    n_fft, hop = 32768, 8192
    S_full = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                             hop_length=hop, n_mels=128, fmax=2000.0)
    S_full_db = librosa.power_to_db(S_full, ref=np.max)

    # Real cepstrum of the representative segment
    window      = np.hanning(len(segment_hp))
    spec        = np.fft.fft(segment_hp * window)
    log_spec    = np.log(np.abs(spec) + 1e-10)
    cepstrum    = np.real(np.fft.ifft(log_spec))
    quefrency   = np.arange(len(cepstrum)) / sr

    return dict(
        y=y, sr=sr, duration=duration, y_hp=y_hp,
        segment=segment, time_seg=time_seg, seg_start_t=seg_start_t,
        S_full_db=S_full_db, hop=hop,
        cepstrum=cepstrum, quefrency=quefrency,
        color=color, n_fft=n_fft,
    )


def draw_column(fig, gs, col, data, title):
    sr = data["sr"]
    color = data["color"]
    min_q_ms, max_q_ms = 0.5, 100.0
    q_min = int(min_q_ms / 1000.0 * sr)
    q_max = int(max_q_ms / 1000.0 * sr)

    # --- Row 0: waveform (full 60 s with highlighted segment) ---
    ax_wave = fig.add_subplot(gs[0, col])
    time_full = np.linspace(0, data["duration"], len(data["y"]))
    ds = 128
    ax_wave.plot(time_full[::ds], data["y"][::ds], color="#5D6D7E", linewidth=0.4, alpha=0.75)
    ax_wave.axvspan(data["seg_start_t"], data["seg_start_t"] + 1.0,
                    color=color, alpha=0.3, label=f"Seg ({data['seg_start_t']:.0f}–{data['seg_start_t']+1:.0f}s)")
    ax_wave.set_title(title, fontsize=11, fontweight="bold", color="#1F3A52")
    ax_wave.set_xlabel("Time (s)", fontsize=9)
    ax_wave.set_ylabel("Amplitude", fontsize=9)
    ax_wave.set_xlim(0, data["duration"])
    ax_wave.grid(True, linestyle="--", alpha=0.4)
    ax_wave.legend(fontsize=8, loc="upper right")

    # --- Row 1: Mel spectrogram ---
    ax_mel = fig.add_subplot(gs[1, col])
    img = librosa.display.specshow(
        data["S_full_db"], sr=sr, hop_length=data["hop"],
        x_axis="time", y_axis="mel", fmax=2000.0,
        ax=ax_mel, cmap="magma", vmin=-80, vmax=-20,
    )
    ax_mel.axvline(data["seg_start_t"] + 0.5, color="white",
                   linestyle="--", linewidth=1.2, alpha=0.7)
    ax_mel.set_title("Mel Spectrogram (0–2 kHz)", fontsize=10, fontweight="bold", color="#1F3A52")
    ax_mel.set_xlabel("Time (s)", fontsize=9)
    ax_mel.set_ylabel("Mel Freq", fontsize=9)
    fig.colorbar(img, ax=ax_mel, pad=0.02).set_label("dBFS", fontsize=8)

    # --- Row 2: Real cepstrum of representative segment ---
    ax_ceps = fig.add_subplot(gs[2, col])
    seg_q_ms = data["quefrency"][q_min:q_max] * 1000.0
    seg_ceps  = data["cepstrum"][q_min:q_max]

    ax_ceps.plot(seg_q_ms, seg_ceps, color=color, linewidth=1.0)

    # Mark top peaks
    q_lf = int(3.0 / 1000.0 * sr)
    pk_lo, _ = find_peaks(data["cepstrum"][q_lf:q_max],
                           distance=int(sr / 1000.0), prominence=0.0005)
    pk_hi, _ = find_peaks(data["cepstrum"][q_min:q_lf],
                           distance=int(sr * 0.1 / 1000.0), prominence=0.0005)
    tops = sorted(zip(data["cepstrum"][pk_hi + q_min],
                       data["quefrency"][pk_hi + q_min] * 1000.0),
                  key=lambda x: x[0], reverse=True)[:1] + \
           sorted(zip(data["cepstrum"][pk_lo + q_lf],
                       data["quefrency"][pk_lo + q_lf] * 1000.0),
                  key=lambda x: x[0], reverse=True)[:4]

    peak_colors = ["#C0392B", "#D35400", "#7D6608", "#273746", "#8E44AD"]
    for i, (val, q_ms) in enumerate(tops):
        f_eq = 1000.0 / q_ms
        ax_ceps.plot(q_ms, val, "o", color=peak_colors[i % 5], markersize=5)
        ax_ceps.annotate(f"{q_ms:.1f}ms\n({f_eq:.0f}Hz)",
                         (q_ms, val), textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=7.5,
                         color=peak_colors[i % 5], fontweight="bold")

    ax_ceps.set_title(f"Real Cepstrum — Seg {data['seg_start_t']:.0f}–{data['seg_start_t']+1:.0f}s",
                      fontsize=10, fontweight="bold", color="#1F3A52")
    ax_ceps.set_xlabel("Quefrency (ms)", fontsize=9)
    ax_ceps.set_ylabel("Cepstral Amplitude", fontsize=9)
    ax_ceps.set_xlim(min_q_ms, max_q_ms)
    ax_ceps.grid(True, linestyle="--", alpha=0.4)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "text.color": "#2C3E50",
        "axes.labelcolor": "#2C3E50",
    })

    print("Processing onset file  (081200)...")
    onset     = process_file(*FILES["onset"][:2], FILES["onset"][2])
    print("Processing peak-pass file (081300)...")
    peak_pass = process_file(*FILES["peak_pass"][:2], FILES["peak_pass"][2])

    fig = plt.figure(figsize=(18, 15), facecolor="#F8F9FA")
    fig.suptitle(
        "Croatia Acoustic Cepstrum — Vessel Onset vs Peak Pass\n"
        "RBW6737_20250725_081200.wav (onset)  vs  RBW6737_20250725_081300.wav (peak pass)",
        fontsize=14, fontweight="bold", y=0.995, color="#1A252C"
    )

    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    draw_column(fig, gs, 0, onset,     "Onset — RBW6737_20250725_081200  (vessel approaching)")
    draw_column(fig, gs, 1, peak_pass, "Peak Pass — RBW6737_20250725_081300  (vessel at closest point)")

    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {os.path.abspath(OUTPUT_IMG)}")


if __name__ == "__main__":
    main()
