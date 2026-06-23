# Underwater Vessel and Soundscape Amplitude Analysis

This workspace contains tools to analyze, segment, and compare underwater acoustic recordings from multiple marine datasets: **Scooter**, **Croatia (Ocean Sonics)**, and **Haifa Bay LME**.

---

## Project Structure

```
rl-vessel-detection/
├── core/                    # Core pipeline modules
│   ├── agent/               #   RL, dispatcher, and tracking agents package
│   │   ├── base_agent.py    #     Base TD learning agent (Q-table, ε-greedy)
│   │   ├── q_learning_agent.py      #     Off-policy Q-Learning
│   │   ├── sarsa_agent.py           #     On-policy SARSA
│   │   ├── double_q_learning_agent.py  #  Double Q-Learning (bias reduction)
│   │   ├── linear_fa_agent.py       #     Linear FA with tile coding
│   │   ├── dyna_q_agent.py          #     Dyna-Q (planning + model)
│   │   ├── dispatcher_agent.py      #     Dispatcher / multi-agent parent
│   │   └── signal_processor_agent.py   #  Child vessel tracking agent
│   ├── environment/         #   Environment abstractions package
│   │   ├── audio_environment.py        #  Async STFT audio streaming
│   │   └── vessel_tracking_rl_env.py   #  MDP wrapper for RL agents
│   └── audio_analyzer.py    #   Segment-level amplitude extractor
├── tools/                   # Utility and plotting scripts
│   ├── plot_architecture.py #  Draw architecture diagrams
│   └── plot_spectrograms.py #  Generate LOFAR/Mel spectrograms
├── run_training.py          # Train a single RL agent
├── run_evaluation.py        # Primary entry point (evaluates an agent)
├── run_orchestrator.py      # Manages parallel evaluation runs
├── run_batch_all.py         # Batch runner for all datasets
├── output/                  # Generated PNGs, CSVs, TXTs (gitignored)
├── requirements.txt
└── README.md
```

> **Configuration**: Set the `RECORDINGS_DIR` environment variable to point to your local recordings root (default: `c:/Users/Roy/Recordings`).

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


### 11. `vessel_detection_timeline.png` & `vessel_detection_report.txt`
* **Generated By**: `vessel_tracker_rl.py` (Dominant Vessel Detection Run)
* **Description**: A visual tracking timeline and matching text report listing dominant tracked vessel frequencies, confidence scores, speed stages, and active durations.
* **Analytical Use**: Provides final validation of the RL agent's vessel detection and speed stage classification performance.