# Underwater Vessel and Soundscape Amplitude Analysis

This workspace contains tools to analyze, segment, and compare underwater acoustic recordings from multiple marine datasets: **Scooter**, **Croatia (Ocean Sonics)**, and **Haifa Bay LME**.

---

## Project Structure

```
rl-vessel-detection/
├── core/                    # Core pipeline modules
│   ├── agent.py             #   RL dispatcher + signal processor agents
│   ├── environment.py       #   Async audio streaming environment
│   └── audio_analyzer.py    #   Segment-level amplitude extractor
├── analysis/                # Named analysis and plotting scripts
│   ├── calculate_segment_distance.py
│   ├── compare_datasets.py
│   ├── compare_distributions_pairwise.py
│   ├── compare_scooter_signatures.py
│   ├── compare_scooter_steady_state_final.py
│   ├── plot_amplitude_histogram.py
│   ├── plot_consecutive_distances.py
│   ├── plot_croatia_cepstrum.py
│   ├── plot_frequency_vs_amplitude.py
│   ├── plot_scooter_cepstrum.py
│   ├── plot_scooter_speeds.py
│   ├── plot_segment_analysis_all.py
│   ├── plot_signal_similarities.py
│   ├── plot_stability_histograms.py
│   └── generate_test_audio.py
├── output/                  # Generated PNGs, CSVs, TXTs (gitignored)
├── vessel_tracker_rl.py     # Primary entry point (RL agent pipeline)
├── vessel_denoiser_nmf.py   # Alternative entry point (denoising NMF pipeline)
├── requirements.txt
└── README.md
```

> **Configuration**: Set the `RECORDINGS_DIR` environment variable to point to your local recordings root (default: `c:/Users/Roy/Recordings` for `vessel_tracker_rl.py` / `D:/RoyStudies/Recordings` for `vessel_denoiser_nmf.py`).

---

## Analysis Pipeline

Follow these steps in order to run the entire analysis pipeline from raw audio recordings to comparative reports:

### Step 1: Extract Segment Amplitudes (1s Non-Overlapping Windows)
Run `audio_analyzer.py` on each target directory to scan audio files and output a segment-by-segment amplitude metrics table (Peak/RMS in linear and dBFS).

```powershell
# 1. Scooter Dataset
.venv\Scripts\python analysis\audio_analyzer.py --dir "C:\Users\Roy\Recordings\scooter" --output output\scooter_amplitude_analysis.csv

# 2. Croatia Ocean Sonics Dataset
.venv\Scripts\python analysis\audio_analyzer.py --dir "C:\Users\Roy\Recordings\Croatia\Ocean Sonics\2507_1" --output output\croatia_amplitude_analysis.csv

# 3. Haifa Bay LME Dataset
.venv\Scripts\python analysis\audio_analyzer.py --dir "C:\Users\Roy\Recordings\20250805_Haifa_bay_LME" --output output\haifa_amplitude_analysis.csv
```

---

### Step 2: Calculate Timeline Distances & Onset Transitions
Run `calculate_segment_distance.py` to parse timestamps from filenames, compute consecutive step transitions (Manhattan distance in dBFS space based on RMS), identify matching pairs, and output distance CSVs.

```powershell
# 1. Scooter Dataset
.venv\Scripts\python analysis\calculate_segment_distance.py --csv output\scooter_amplitude_analysis.csv --output-csv output\scooter_segment_distances.csv --metric manhattan_db --feature rms

# 2. Croatia Dataset
.venv\Scripts\python analysis\calculate_segment_distance.py --csv output\croatia_amplitude_analysis.csv --output-csv output\croatia_segment_distances.csv --metric manhattan_db --feature rms

# 3. Haifa Bay Dataset (if Haifa amplitude analysis CSV exists)
.venv\Scripts\python analysis\calculate_segment_distance.py --csv output\haifa_amplitude_analysis.csv --output-csv output\haifa_segment_distances.csv --metric manhattan_db --feature rms
```

---

### Step 3: Run Comparative Analysis
Generate the comparative statistical report and boxplot/CDF/histogram grids comparing the overall soundscapes. The Haifa Bay dataset argument is optional.

```powershell
.venv\Scripts\python analysis\compare_datasets.py --scooter output\scooter_segment_distances.csv --croatia output\croatia_segment_distances.csv --output-img output\dataset_comparison_report.png
```

---

### Step 4: Extract and Verify Steady-State Scooter Motor Hum
Run the transient-filtering fingerprint script to isolate the actual continuous scooter sound signal from background spikes and clipping/saturation clicks using rolling median filters, and output comparison stats.

```powershell
.venv\Scripts\python analysis\compare_scooter_steady_state_final.py
```
Output is saved to `output\scooter_steady_state_comparison.png`.

---

### Step 5: Generate Visualizations & Consolidated Plots
Generate the consolidated plots and specialized visualization charts.

```powershell
# Consolidated side-by-side amplitude histograms (Scooter vs Croatia)
.venv\Scripts\python analysis\plot_amplitude_histogram.py

# Consolidated consecutive distance timeline transitions
.venv\Scripts\python analysis\plot_consecutive_distances.py

# Two-column cepstrum comparison (Croatia onset vs peak pass)
.venv\Scripts\python analysis\plot_croatia_cepstrum.py

# Scooter motor hum cepstrum
.venv\Scripts\python analysis\plot_scooter_cepstrum.py

# Joint frequency vs amplitude distribution
.venv\Scripts\python analysis\plot_frequency_vs_amplitude.py

# Speed-binned stability metrics
.venv\Scripts\python analysis\plot_stability_histograms.py

# Full speed profile timeline
.venv\Scripts\python analysis\plot_scooter_speeds.py

# Summary grid of Croatia segment analyses
.venv\Scripts\python analysis\plot_segment_analysis_all.py
```
All outputs are saved to the `output/` directory.

---

## Overview of Generated Plots

Here is a summary of all figures generated in the `output/` directory, detailing what is plotted and its analytical purpose:

### 1. `amplitude_histograms.png`
* **Generated By**: `analysis/plot_amplitude_histogram.py`
* **Description**: Side-by-side probability density histograms comparing the Scooter and Croatia datasets.
* **Key Panels**:
  * **Left Panel**: Distribution of Root-Mean-Square (RMS) loudness values (dBFS).
  * **Right Panel**: Distribution of Peak amplitude values (dBFS).
* **Analytical Use**: Visualizes overall loudness ranges, acoustic floors, variance, and mean offsets between the datasets.

### 2. `consecutive_distances.png`
* **Generated By**: `analysis/plot_consecutive_distances.py`
* **Description**: A stacked panel timeline plot showing the Manhattan dB distance between consecutive 1-second segment features for both datasets over time.
* **Key Panels**:
  * **Top Panel**: Consecutive segment transition timeline for the Scooter dataset.
  * **Bottom Panel**: Consecutive segment transition timeline for the Croatia dataset.
* **Analytical Use**: Highlights periods of high acoustic volatility or transient spikes (shaded in red when crossing the 10 dB event transition threshold).

### 3. `croatia_cepstrum_comparison.png`
* **Generated By**: `analysis/plot_croatia_cepstrum.py`
* **Description**: A detailed two-column comparative analysis of Croatia recordings before and during a vessel pass.
* **Key Panels**:
  * **Left Column**: Vessel Onset (vessel approaching).
  * **Right Column**: Vessel Peak Pass (vessel at the closest point).
  * **Subplots**: Each column plots the raw time-domain waveform, the 0-2 kHz Mel Spectrogram, and the Real Cepstrum of a representative 1-second segment with identified harmonics annotated.
* **Analytical Use**: Displays how harmonic structure and peak frequencies change as the vessel gets closer.

### 4. `scooter_cepstrum_analysis.png`
* **Generated By**: `analysis/plot_scooter_cepstrum.py`
* **Description**: Plots waveforms, spectrograms, cepstrum timelines, and peak patterns for the Scooter dataset.
* **Analytical Use**: Specifically tuned to identify the harmonic "comb" signature of a scooter motor hum.

### 5. `scooter_freq_vs_amplitude.png`
* **Generated By**: `analysis/plot_frequency_vs_amplitude.py`
* **Description**: A 2D joint distribution plot comparing signal peak frequency versus amplitude.
* **Analytical Use**: Identifies dominant engine frequencies across different speed/loudness regimes.

### 6. `scooter_stability_histograms.png`
* **Generated By**: `analysis/plot_stability_histograms.py`
* **Description**: Histograms showing speed-binned stability metrics of the tracked engine sounds.
* **Analytical Use**: Verifies how steadily the tracking agent holds onto target frequency components across speeds.

### 7. `scooter_speeds_profile.png`
* **Generated By**: `analysis/plot_scooter_speeds.py`
* **Description**: Timelines mapping tracked speeds and frequency values across all scooter recordings.
* **Analytical Use**: Provides an overall speed profile timeline of the vessel during the session.

### 8. `segment_analysis_summary.png`
* **Generated By**: `analysis/plot_segment_analysis_all.py`
* **Description**: A comprehensive grid plotting segment analysis results across all Croatia subfolders.
* **Analytical Use**: Compares transient event densities and average sound level behavior across folders.

### 9. `dataset_comparison_report.png`
* **Generated By**: `analysis/compare_datasets.py`
* **Description**: A four-panel comparative report containing overlapping histograms, boxplots of RMS spread, boxplots of consecutive transitions, and a cumulative distribution function (CDF) curve.
* **Analytical Use**: Performs overall soundscape shape and level comparisons between Scooter and Croatia datasets.

### 10. `scooter_steady_state_comparison.png`
* **Generated By**: `analysis/compare_scooter_steady_state_final.py`
* **Description**: Shows the statistical distribution fingerprint comparisons using a Z-score normalized Kolmogorov-Smirnov test.
* **Analytical Use**: Compares the steady-state motor hum of the scooter after filtering out transients against background noise.

### 11. `vessel_detection_timeline.png` & `vessel_detection_report.txt`
* **Generated By**: `vessel_tracker_rl.py` (Dominant Vessel Detection Run)
* **Description**: A visual tracking timeline and matching text report listing dominant tracked vessel frequencies, confidence scores, speed stages, and active durations.
* **Analytical Use**: Provides final validation of the RL agent's vessel detection and speed stage classification performance.