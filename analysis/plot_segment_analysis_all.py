"""
plot_segment_analysis_all.py
============================
Runs the 1-second segment analysis for:
  • The Scooter reference dataset
  • Every immediate subdirectory under Croatia/Ocean Sonics

For each subdirectory it produces one PNG:
    segment_analysis_<subfolder>.png

Then produces a combined SUMMARY figure comparing key statistics across
all subdirectories side-by-side.

Output files are saved both in the workspace and in the brain artifacts dir.
"""

import os
import glob
import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, lfilter

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "#F4F6F9",
})

# ── Constants ──────────────────────────────────────────────────────────────────
SR        = 16_000
SEG_SEC   = 1.0
HP_CUT    = 150.0
BAND_LO   = 580.0
BAND_HI   = 950.0
AMP_FLOOR = -95.0
GEAR_SPLIT = 750.0      # < 750 Hz → Low Speed, >= 750 Hz → High Speed

C_AMP   = "#27AE60"
C_DELTA = "#8E44AD"
BG_W    = "#FFFFFF"

# A pool of colours for the subdirectories
SUBDIR_COLORS = [
    "#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#E67E22",
    "#16A085", "#D35400", "#2C3E50", "#C0392B", "#1ABC9C",
]


# ── DSP ───────────────────────────────────────────────────────────────────────
def hp_filter(y, cutoff=HP_CUT, fs=SR, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high", analog=False)
    return lfilter(b, a, y)


def extract_segments(filepath):
    y, _ = librosa.load(filepath, sr=SR, mono=True)
    if len(y) < SR:
        return []
    y_hp = hp_filter(y)
    wlen = int(SEG_SEC * SR)
    segs = []
    for i in range(len(y_hp) // wlen):
        seg = y_hp[i * wlen: (i + 1) * wlen]
        if np.std(seg) < 1e-10:
            continue
        windowed = seg * np.hanning(wlen)
        spec   = np.fft.rfft(windowed)
        f_axis = np.fft.rfftfreq(wlen, 1.0 / SR)
        mag    = np.abs(spec)
        band   = np.where((f_axis >= BAND_LO) & (f_axis <= BAND_HI))[0]
        if not len(band):
            continue
        pi = band[np.argmax(mag[band])]
        hz = f_axis[pi]
        db = 20.0 * np.log10(mag[pi] / (wlen / 2.0) + 1e-10)
        if db > AMP_FLOOR:
            segs.append((hz, db))
    return segs


def load_dataset(directory, tag=""):
    files  = sorted(glob.glob(os.path.join(directory, "*.wav")))
    hz_l, db_l, fi_l, starts = [], [], [], []
    for fi, fp in enumerate(files):
        starts.append(len(hz_l))
        for hz, db in extract_segments(fp):
            hz_l.append(hz); db_l.append(db); fi_l.append(fi)
        if tag and ((fi + 1) % 10 == 0 or fi == len(files) - 1):
            print(f"    [{fi+1:3d}/{len(files)}] {os.path.basename(fp)}")
    names = [os.path.basename(f) for f in files]
    return (np.asarray(hz_l, float), np.asarray(db_l, float),
            np.asarray(fi_l, int), names, starts)


def load_dataset_full(directory):
    """
    Like load_dataset but keeps EVERY 1-second window without any amplitude
    floor filter.  Frequency is stored as NaN for windows where the peak in
    [BAND_LO, BAND_HI] is below AMP_FLOOR (= scooter not detectable).

    This lets us plot the FULL recording timeline — showing when the scooter
    approaches (amplitude rises), is audible, and leaves (amplitude falls).

    Returns
    -------
    times    : (N,) float  – segment start time in seconds from recording start
    hz_full  : (N,) float  – peak Hz (NaN when below AMP_FLOOR)
    db_full  : (N,) float  – peak dBFS in band (always present)
    starts_t : list[float] – absolute start-time (s) of each file
    """
    files    = sorted(glob.glob(os.path.join(directory, "*.wav")))
    times, hz_full, db_full = [], [], []
    starts_t = []
    t_offset = 0.0
    wlen     = int(SEG_SEC * SR)

    for fp in files:
        starts_t.append(t_offset)
        y, _ = librosa.load(fp, sr=SR, mono=True)
        y_hp = hp_filter(y)
        n_wins = len(y_hp) // wlen
        for i in range(n_wins):
            seg = y_hp[i * wlen: (i + 1) * wlen]
            t   = t_offset + i * SEG_SEC
            times.append(t)
            if np.std(seg) < 1e-10:
                hz_full.append(np.nan); db_full.append(np.nan)
                continue
            windowed = seg * np.hanning(wlen)
            spec   = np.fft.rfft(windowed)
            f_ax   = np.fft.rfftfreq(wlen, 1.0 / SR)
            mag    = np.abs(spec)
            band   = np.where((f_ax >= BAND_LO) & (f_ax <= BAND_HI))[0]
            if not len(band):
                hz_full.append(np.nan); db_full.append(np.nan)
                continue
            pi   = band[np.argmax(mag[band])]
            db_v = 20.0 * np.log10(mag[pi] / (wlen / 2.0) + 1e-10)
            db_full.append(db_v)
            hz_full.append(f_ax[pi] if db_v > AMP_FLOOR else np.nan)
        t_offset += len(y) / SR    # actual duration (handles short last file)

    return (np.asarray(times, float),
            np.asarray(hz_full, float),
            np.asarray(db_full, float),
            starts_t)


# ── Deltas + Mahalanobis distance ───────────────────────────────────────────────────
def compute_deltas(hz, db, fi):
    """
    For each consecutive pair *within the same file* compute:
      delta_hz   – signed frequency change (Hz)
      delta_db   – signed amplitude change (dBFS)
      d_mahal    – Mahalanobis distance in (hz, db) feature space

    Mahalanobis distance is FAIR: it normalises each axis by its own
    standard deviation AND accounts for the hz–amplitude correlation,
    so a 1-sigma jump in frequency equals a 1-sigma jump in amplitude.

    Gaps between files are stored as NaN.
    """
    n   = len(hz)
    dhz = np.full(n, np.nan)
    ddb = np.full(n, np.nan)
    dm  = np.full(n, np.nan)

    # Build the 2-D covariance matrix from ALL (hz, db) pairs
    # so the metric reflects the natural spread of this specific dataset.
    if n >= 2:
        data = np.column_stack([hz, db])          # shape (N, 2)
        cov  = np.cov(data.T)                     # 2x2 covariance matrix
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(2)                   # fallback: identity (= Euclidean)
    else:
        cov_inv = np.eye(2)

    for i in range(n - 1):
        if fi[i] != fi[i + 1]:                    # file boundary – skip
            continue
        dhz[i] = hz[i + 1] - hz[i]
        ddb[i] = db[i + 1] - db[i]
        delta  = np.array([dhz[i], ddb[i]])
        dm[i]  = np.sqrt(max(delta @ cov_inv @ delta, 0.0))

    return dhz, ddb, dm


# ── Quick stats dict ──────────────────────────────────────────────────────────
def dataset_stats(hz, db, dhz, ddb, dm, n_files):
    v = ~np.isnan(dhz)
    return dict(
        n_files     = n_files,
        n_segs      = len(hz),
        freq_mean   = np.mean(hz)    if len(hz) else np.nan,
        freq_std    = np.std(hz)     if len(hz) else np.nan,
        amp_mean    = np.mean(db)    if len(db) else np.nan,
        amp_std     = np.std(db)     if len(db) else np.nan,
        dhz_mean    = np.mean(np.abs(dhz[v])) if v.any() else np.nan,
        pct_stable_hz = np.mean(np.abs(dhz[v]) < 20) * 100 if v.any() else np.nan,
        ddb_mean    = np.mean(np.abs(ddb[v])) if v.any() else np.nan,
        mahal_mean  = np.nanmean(dm) if len(dm) else np.nan,
    )


# ── Per-subdirectory figure ───────────────────────────────────────────────────
def plot_single(sc_data, cr_data, cr_full_data, cr_label, color, out_path_ws, out_path_art):
    sc_hz, sc_db, sc_fi, sc_names, sc_starts = sc_data
    cr_hz, cr_db, cr_fi, cr_names, cr_starts = cr_data

    sc_dhz, sc_ddb, sc_dl2 = compute_deltas(sc_hz, sc_db, sc_fi)
    cr_dhz, cr_ddb, cr_dl2 = compute_deltas(cr_hz, cr_db, cr_fi)

    SC_COLOR = "#E74C3C"

    fig = plt.figure(figsize=(22, 34))
    fig.suptitle(
        f"1-Second Segment Analysis  —  Scooter  vs  Croatia / {cr_label}\n"
        f"Scooter: {len(sc_names)} files · {len(sc_hz):,} segs     |     "
        f"Croatia {cr_label}: {len(cr_names)} files · {len(cr_hz):,} segs  "
        f"(full timeline: {len(cr_full_data[0]):,} windows)",
        fontsize=13, fontweight="bold", y=0.995, color="#1A252C"
    )

    gs = fig.add_gridspec(
        6, 2,
        height_ratios=[0.85, 0.85, 1.1, 1.1, 0.85, 1.1],
        hspace=0.52, wspace=0.32,
        left=0.07, right=0.97, top=0.97, bottom=0.03,
    )

    # ── Row 0: Frequency histograms ──────────────────────────────────────────
    for col, (hz, clr, title) in enumerate([
        (sc_hz, SC_COLOR, f"Scooter ({len(sc_names)} files, {len(sc_hz):,} segs)  — Freq"),
        (cr_hz, color,   f"Croatia {cr_label} ({len(cr_names)} files, {len(cr_hz):,} segs) — Freq"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        bins = np.arange(BAND_LO, BAND_HI + 5, 5)
        ax.hist(hz, bins=bins, color=clr, alpha=0.72, rwidth=0.87, edgecolor="white")
        mu, std = np.mean(hz), np.std(hz)
        ax.axvline(mu, color=clr, lw=2.0, ls="-", alpha=0.9)
        ax.axvspan(mu - std, mu + std, alpha=0.12, color=clr)
        ax.axvline(630, color="#666", lw=1.2, ls=":", alpha=0.7)
        ax.axvline(856, color="#666", lw=1.2, ls=":", alpha=0.7)
        ymax = ax.get_ylim()[1]
        ax.text(634, ymax * 0.97, "630 Hz\n(Low)",  fontsize=8, color="#555", va="top")
        ax.text(860, ymax * 0.97, "856 Hz\n(High)", fontsize=8, color="#555", va="top")
        ax.text(0.97, 0.96, f"n={len(hz):,}\nmean={mu:.1f} Hz\nstd={std:.1f} Hz",
                transform=ax.transAxes, fontsize=9, va="top", ha="right", color=clr,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=clr, alpha=0.88))
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1F3A52")
        ax.set_xlabel("Peak Frequency (Hz)", fontsize=10)
        ax.set_ylabel("Segment Count",       fontsize=10)
        ax.set_xlim(BAND_LO, BAND_HI)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_facecolor(BG_W)

    # ── Row 1: Amplitude histograms ──────────────────────────────────────────
    for col, (db, clr, title) in enumerate([
        (sc_db, SC_COLOR, "Scooter — Amplitude"),
        (cr_db, color,   f"Croatia {cr_label} — Amplitude"),
    ]):
        ax = fig.add_subplot(gs[1, col])
        bins = np.arange(AMP_FLOOR, -13, 2)
        ax.hist(db, bins=bins, color=clr, alpha=0.72, rwidth=0.87, edgecolor="white")
        mu, std = np.mean(db), np.std(db)
        ax.axvline(mu, color=clr, lw=2.0, ls="-", alpha=0.9)
        ax.axvspan(mu - std, mu + std, alpha=0.12, color=clr)
        ax.text(0.03, 0.96, f"n={len(db):,}\nmean={mu:.1f} dBFS\nstd={std:.1f} dBFS",
                transform=ax.transAxes, fontsize=9, va="top", ha="left", color=clr,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=clr, alpha=0.88))
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1F3A52")
        ax.set_xlabel("Peak Amplitude (dBFS)", fontsize=10)
        ax.set_ylabel("Segment Count",         fontsize=10)
        ax.set_xlim(AMP_FLOOR, -13)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_facecolor(BG_W)

    # ── Row 2: Timelines ─────────────────────────────────────────────────────
    for col, (hz, db, fi, starts, clr, title) in enumerate([
        (sc_hz, sc_db, sc_fi, sc_starts, SC_COLOR, "Scooter — Segment Timeline"),
        (cr_hz, cr_db, cr_fi, cr_starts, color,   f"Croatia {cr_label} — Segment Timeline"),
    ]):
        ax  = fig.add_subplot(gs[2, col])
        ax2 = ax.twinx()
        x   = np.arange(len(hz))
        lo  = hz < GEAR_SPLIT
        ax.scatter(x[lo],  hz[lo],  s=5, color=clr,    alpha=0.55, linewidths=0, label="Low (<750 Hz)")
        ax.scatter(x[~lo], hz[~lo], s=5, color=C_DELTA, alpha=0.55, linewidths=0, label="High (>=750 Hz)")
        ax2.plot(x, db, color=C_AMP, lw=0.7, alpha=0.45)
        for s in starts[1:]:
            ax.axvline(s, color="#BDC3C7", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(630, color="#999", lw=0.8, ls=":", alpha=0.7)
        ax.axhline(856, color="#999", lw=0.8, ls=":", alpha=0.7)
        ax.text(1, 633, "630 Hz", fontsize=7.5, color="#888")
        ax.text(1, 859, "856 Hz", fontsize=7.5, color="#888")
        ax.set_ylim(BAND_LO - 20, BAND_HI + 30)
        ax2.set_ylim(AMP_FLOOR - 5, -10)
        ax.set_xlim(0, max(len(x) - 1, 1))
        ax.set_title(f"{title}\n(dots=freq by regime, green=amp, dashes=file boundaries)",
                     fontsize=9.5, fontweight="bold", color="#1F3A52")
        ax.set_xlabel("Segment index (1 seg = 1 s)", fontsize=9)
        ax.set_ylabel("Peak Frequency (Hz)",          fontsize=9, color=clr)
        ax2.set_ylabel("Peak Amplitude (dBFS)",        fontsize=9, color=C_AMP)
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.set_facecolor(BG_W)
        handles = [
            mpatches.Patch(color=clr,    label="Low speed (<750 Hz)"),
            mpatches.Patch(color=C_DELTA, label="High speed (>=750 Hz)"),
            mpatches.Patch(color=C_AMP,   label="Amplitude (dBFS)"),
        ]
        ax.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.85)

    # ── Row 3: Consecutive delta ─────────────────────────────────────────────
    for col, (dhz, ddb, clr, title) in enumerate([
        (sc_dhz, sc_ddb, SC_COLOR, "Scooter — Consecutive Segment Change"),
        (cr_dhz, cr_ddb, color,   f"Croatia {cr_label} — Consecutive Segment Change"),
    ]):
        ax  = fig.add_subplot(gs[3, col])
        ax2 = ax.twinx()
        x    = np.arange(len(dhz))
        valid = ~np.isnan(dhz)
        pm = valid & (dhz >= 0)
        nm = valid & (dhz < 0)
        if pm.any():
            ax.bar(x[pm], dhz[pm], width=1, color=clr, alpha=0.65, linewidth=0)
        if nm.any():
            ax.bar(x[nm], dhz[nm], width=1, color=clr, alpha=0.38, linewidth=0)
        ax.axhline(0, color="#888", lw=0.8)
        ax2.plot(x, ddb, color=C_AMP, lw=0.8, alpha=0.65)
        ax2.axhline(0, color=C_AMP, lw=0.5, ls="--", alpha=0.5)
        if valid.any():
            vhz = np.abs(dhz[valid]); vdb = np.abs(ddb[valid])
            pct_hz = np.mean(vhz < 20) * 100
            pct_db = np.mean(vdb < 3)  * 100
            ax.text(0.97, 0.97,
                    f"|dHz|<20 in {pct_hz:.0f}% segs\n"
                    f"mean|dHz|={np.mean(vhz):.1f} Hz\n"
                    f"|dAmp|<3 in {pct_db:.0f}% segs\n"
                    f"mean|dAmp|={np.mean(vdb):.1f} dBFS",
                    transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=clr, alpha=0.90))
        ax.set_title(f"{title}\n(bars=delta Hz, green=delta dBFS)",
                     fontsize=9.5, fontweight="bold", color="#1F3A52")
        ax.set_xlabel("Segment index",         fontsize=9)
        ax.set_ylabel("Delta Frequency (Hz)",  fontsize=9, color=clr)
        ax2.set_ylabel("Delta Amplitude (dBFS)", fontsize=9, color=C_AMP)
        ax.set_xlim(0, max(len(dhz) - 1, 1))
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.set_facecolor(BG_W)

    # ── Row 4: L2 distance ───────────────────────────────────────────────────
    ax_l2 = fig.add_subplot(gs[4, :])
    sc_x  = np.arange(len(sc_dl2))
    cr_x  = np.arange(len(cr_dl2))
    sv    = ~np.isnan(sc_dl2)
    cv    = ~np.isnan(cr_dl2)
    offset = len(sc_dl2) + 20
    ax_l2.plot(sc_x[sv], sc_dl2[sv], color=SC_COLOR, lw=0.9, alpha=0.8,
               label=f"Scooter (mean={np.nanmean(sc_dl2):.3f}, n={sv.sum():,})")
    ax_l2.plot(cr_x[cv] + offset, cr_dl2[cv], color=color, lw=0.9, alpha=0.8,
               label=f"Croatia {cr_label} (mean={np.nanmean(cr_dl2):.3f}, n={cv.sum():,})")
    ax_l2.axvline(offset - 10, color="#AAB7B8", lw=1.8, ls="--")
    ax_l2.axhline(np.nanmean(sc_dl2), color=SC_COLOR, lw=1.0, ls="--", alpha=0.6)
    ax_l2.axhline(np.nanmean(cr_dl2), color=color,    lw=1.0, ls="--", alpha=0.6)
    ax_l2.set_title(
        "Mahalanobis Distance Between Consecutive 1-Second Segments\n"
        "d = sqrt( delta^T * S^-1 * delta )  where S = dataset covariance matrix  |  "
        "FAIR: 1-sigma change in freq = 1-sigma change in amplitude",
        fontsize=10, fontweight="bold", color="#1F3A52"
    )
    ax_l2.set_xlabel("Segment index (Scooter first, then Croatia)", fontsize=10)
    ax_l2.set_ylabel("Mahalanobis Distance", fontsize=10)
    ax_l2.legend(fontsize=9.5, loc="upper right")
    ax_l2.grid(ls="--", alpha=0.40)
    ax_l2.set_facecolor(BG_W)
    ax_l2.set_xlim(0, offset + len(cr_dl2))

    # ── Row 5: Full Croatia timeline (ALL windows, no amplitude floor) ────────
    cr_times, cr_hz_full, cr_db_full, cr_starts_t = cr_full_data
    t_min = cr_times / 60.0    # convert seconds → minutes
    n_total   = len(cr_times)
    n_detect  = int(np.sum(~np.isnan(cr_hz_full)))
    pct_det   = 100.0 * n_detect / max(n_total, 1)

    ax_full  = fig.add_subplot(gs[5, :])
    ax_full2 = ax_full.twinx()

    # Amplitude line (always) – continuous, shows approach/departure
    ax_full2.plot(t_min, cr_db_full, color=C_AMP, lw=0.6, alpha=0.55,
                  label="Amplitude (dBFS) — all windows")
    ax_full2.axhline(AMP_FLOOR, color="#E74C3C", lw=1.2, ls="--", alpha=0.8,
                     label=f"Detection threshold ({AMP_FLOOR} dBFS)")
    # Shade below-threshold region
    ax_full2.fill_between(t_min, cr_db_full, AMP_FLOOR,
                           where=cr_db_full < AMP_FLOOR,
                           color="#E74C3C", alpha=0.08, label="Below threshold")

    # Frequency scatter – only where detected
    det_mask = ~np.isnan(cr_hz_full)
    lo_mask  = det_mask & (cr_hz_full < GEAR_SPLIT)
    hi_mask  = det_mask & (cr_hz_full >= GEAR_SPLIT)
    ax_full.scatter(t_min[lo_mask], cr_hz_full[lo_mask], s=4,
                    color=color,   alpha=0.65, linewidths=0, label="Low speed (<750 Hz)")
    ax_full.scatter(t_min[hi_mask], cr_hz_full[hi_mask], s=4,
                    color=C_DELTA, alpha=0.65, linewidths=0, label="High speed (>=750 Hz)")

    # File boundary marks every ~60 s (thin vertical lines)
    for st in cr_starts_t[1:]:
        ax_full.axvline(st / 60.0, color="#BDC3C7", lw=0.5, ls="--", alpha=0.4)

    # Reference Hz guides
    ax_full.axhline(630, color="#999", lw=0.8, ls=":", alpha=0.7)
    ax_full.axhline(856, color="#999", lw=0.8, ls=":", alpha=0.7)
    ax_full.text(t_min[1] if len(t_min) > 1 else 0, 633,
                 "630 Hz", fontsize=7.5, color="#888")
    ax_full.text(t_min[1] if len(t_min) > 1 else 0, 860,
                 "856 Hz", fontsize=7.5, color="#888")

    ax_full.set_ylim(BAND_LO - 20, BAND_HI + 50)
    ax_full2.set_ylim(min(np.nanmin(cr_db_full) - 5, AMP_FLOOR - 10), -10)
    ax_full.set_xlim(t_min[0] if len(t_min) else 0,
                     t_min[-1] if len(t_min) else 1)

    ax_full.set_title(
        f"Croatia {cr_label} — FULL Recording Timeline  "
        f"({n_total:,} windows = {n_total/60:.0f} min total  |  "
        f"{n_detect:,} windows above detection threshold = {pct_det:.1f}%)\n"
        "Gaps = scooter not detectable (below noise floor in 580-950 Hz band)  "
        "|  Amplitude line shows vessel approach and departure",
        fontsize=10, fontweight="bold", color="#1F3A52"
    )
    ax_full.set_xlabel("Recording time (minutes)", fontsize=10)
    ax_full.set_ylabel("Detected Frequency (Hz)",  fontsize=10, color=color)
    ax_full2.set_ylabel("Peak Amplitude in band (dBFS)", fontsize=10, color=C_AMP)
    ax_full.grid(axis="y", ls="--", alpha=0.3)
    ax_full.set_facecolor(BG_W)

    handles = [
        mpatches.Patch(color=color,   label=f"Low speed (<750 Hz)  detected"),
        mpatches.Patch(color=C_DELTA,  label=f"High speed (>=750 Hz) detected"),
        mpatches.Patch(color=C_AMP,    label="Amplitude (all windows)"),
        mpatches.Patch(color="#E74C3C", label=f"Detection threshold ({AMP_FLOOR} dBFS)"),
    ]
    ax_full.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.88)

    plt.savefig(out_path_ws,  dpi=130, bbox_inches="tight")
    plt.savefig(out_path_art, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path_ws}")

    return (compute_deltas(sc_hz, sc_db, sc_fi),
            compute_deltas(cr_hz, cr_db, cr_fi),
            len(sc_names), len(cr_names))


# ── Summary figure ─────────────────────────────────────────────────────────────
def plot_summary(sc_stats, all_cr_stats, subdirs, colors, out_ws, out_art):
    """
    Bar chart comparison across all subdirectories for:
      - mean freq, freq std, % stable Hz, mean |dHz|, mean L2 distance
    """
    labels = subdirs
    n      = len(labels)
    x      = np.arange(n)
    w      = 0.35

    metrics = [
        ("freq_mean",     "Mean Frequency (Hz)",               "Hz"),
        ("freq_std",      "Frequency Std Dev (Hz)",             "Hz"),
        ("pct_stable_hz", "% Segments Stable (|dHz|<20)",      "%"),
        ("dhz_mean",      "Mean |Delta Freq| (Hz)",             "Hz"),
        ("mahal_mean",    "Mean Mahalanobis Distance\n(fair: 1-sigma freq = 1-sigma amp)", ""),
        ("amp_mean",      "Mean Amplitude (dBFS)",              "dBFS"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Summary Comparison — Scooter vs All Croatia Subdirectories\n"
        "Each metric shows how stable and consistent the motor-hum signal is per subdirectory",
        fontsize=13, fontweight="bold", color="#1A252C"
    )
    axes = axes.flatten()

    SC_COLOR = "#E74C3C"

    for ax_i, (key, title, unit) in enumerate(metrics):
        ax = axes[ax_i]
        sc_val = sc_stats[key]

        cr_vals = [s[key] for s in all_cr_stats]
        bars_cr = ax.bar(x, cr_vals, width=w * 1.6, color=colors[:n], alpha=0.72,
                         edgecolor="white", linewidth=0.5, label="Croatia subdir")

        # Scooter as a horizontal dashed reference line
        ax.axhline(sc_val, color=SC_COLOR, lw=2.0, ls="--",
                   label=f"Scooter ref = {sc_val:.2f}")

        # Value labels on bars
        for bar, val in zip(bars_cr, cr_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + abs(bar.get_height()) * 0.01,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#1F3A52")
        ax.set_ylabel(unit, fontsize=10)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_facecolor(BG_W)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_ws,  dpi=150, bbox_inches="tight")
    plt.savefig(out_art, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary saved -> {out_ws}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    SCOOTER_DIR   = r"C:\Users\Roy\Recordings\scooter"
    OCEAN_SONICS  = r"C:\Users\Roy\Recordings\Croatia\Ocean Sonics"
    ARTIFACT_DIR  = "output"

    # ── Discover subdirectories ─────────────────────────────────────────────
    subdirs = sorted([
        d for d in os.listdir(OCEAN_SONICS)
        if os.path.isdir(os.path.join(OCEAN_SONICS, d))
        and glob.glob(os.path.join(OCEAN_SONICS, d, "*.wav"))
    ])
    print(f"Found {len(subdirs)} Croatia subdirectories: {subdirs}\n")

    # ── Load Scooter once (shared reference) ────────────────────────────────
    print("Loading Scooter reference dataset ...")
    sc_data = load_dataset(SCOOTER_DIR, tag="Scooter")
    sc_hz, sc_db, sc_fi, sc_names, sc_starts = sc_data
    sc_dhz, sc_ddb, sc_dl2 = compute_deltas(sc_hz, sc_db, sc_fi)
    sc_stats = dataset_stats(sc_hz, sc_db, sc_dhz, sc_ddb, sc_dl2, len(sc_names))
    print(f"  => {len(sc_hz):,} segments from {len(sc_names)} files\n")

    # ── Process each subdirectory ────────────────────────────────────────────
    all_cr_stats = []

    for ci, subdir in enumerate(subdirs):
        color     = SUBDIR_COLORS[ci % len(SUBDIR_COLORS)]
        subdir_path = os.path.join(OCEAN_SONICS, subdir)

        print(f"[{ci+1}/{len(subdirs)}] Processing Croatia/{subdir} ...")
        cr_data = load_dataset(subdir_path, tag=subdir)
        cr_hz, cr_db, cr_fi, cr_names, cr_starts = cr_data
        cr_dhz, cr_ddb, cr_dl2 = compute_deltas(cr_hz, cr_db, cr_fi)
        cr_stats = dataset_stats(cr_hz, cr_db, cr_dhz, cr_ddb, cr_dl2, len(cr_names))
        all_cr_stats.append(cr_stats)

        print(f"  => {len(cr_hz):,} segments from {len(cr_names)} files")
        v = ~np.isnan(cr_dhz)
        print(f"     freq {np.mean(cr_hz):.1f} Hz  std {np.std(cr_hz):.1f} Hz  "
              f"mean|dHz| {np.mean(np.abs(cr_dhz[v])):.1f}  Mahal {np.nanmean(cr_dl2):.4f}")

        print(f"  Loading full timeline ({subdir}) ...")
        cr_full_data = load_dataset_full(subdir_path)
        n_full = len(cr_full_data[0])
        n_det  = int(np.sum(~np.isnan(cr_full_data[1])))
        print(f"     Full: {n_full:,} windows, {n_det:,} above floor ({100*n_det/max(n_full,1):.1f}%)")

        out_ws  = f"segment_analysis_{subdir}.png"
        out_art = os.path.join(ARTIFACT_DIR, out_ws)

        plot_single(sc_data, cr_data, cr_full_data, subdir, color, out_ws, out_art)
        print()

    # ── Summary figure ───────────────────────────────────────────────────────
    print("Generating summary comparison figure ...")
    colors_used = [SUBDIR_COLORS[i % len(SUBDIR_COLORS)] for i in range(len(subdirs))]
    plot_summary(
        sc_stats, all_cr_stats, subdirs, colors_used,
        out_ws  = "segment_analysis_summary.png",
        out_art = os.path.join(ARTIFACT_DIR, "segment_analysis_summary.png"),
    )

    print("\nAll done!")
    print("Per-subdirectory plots:")
    for subdir in subdirs:
        print(f"  segment_analysis_{subdir}.png")
    print("Summary: segment_analysis_summary.png")


if __name__ == "__main__":
    main()
