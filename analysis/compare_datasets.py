import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_dataset_metrics(csv_path):
    """
    Loads segment loudness and transition distance metrics from a CSV file.
    """
    rms_db = []
    transitions = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rms = float(row["RMS_dBFS"])
                dist = float(row["Distance_to_Next"])
                
                # Check for silence fallback (-100 dBFS)
                rms_db.append(rms)
                if dist > 0.0:  # ignore 0.0 at file boundaries
                    transitions.append(dist)
            except (ValueError, KeyError):
                continue
                
    return np.array(rms_db), np.array(transitions)

def main():
    parser = argparse.ArgumentParser(description="Compare Scooter, Croatia, and Haifa Bay acoustic datasets.")
    parser.add_argument("--scooter", default="scooter_segment_distances.csv", help="Scooter dataset CSV path")
    parser.add_argument("--croatia", default="croatia_segment_distances.csv", help="Croatia dataset CSV path")
    parser.add_argument("--haifa", default="haifa_segment_distances.csv", help="Haifa Bay dataset CSV path")
    parser.add_argument("--output-img", default="dataset_comparison_report.png", help="Comparison plot filename")
    parser.add_argument("--extra-dir", default=None, help="Directory to save an extra copy of the plot")
    parser.add_argument("--threshold", type=float, default=10.0, help="Event transition threshold in dB (default: 10.0)")
    
    args = parser.parse_args()
    
    try:
        scooter_rms, scooter_trans = load_dataset_metrics(args.scooter)
        print(f"Loaded Scooter dataset: {len(scooter_rms)} segments.")
    except Exception as e:
        print(f"Error loading Scooter data: {e}")
        return
        
    try:
        croatia_rms, croatia_trans = load_dataset_metrics(args.croatia)
        print(f"Loaded Croatia dataset: {len(croatia_rms)} segments.")
    except Exception as e:
        print(f"Error loading Croatia data: {e}")
        return

    has_haifa = False
    try:
        haifa_rms, haifa_trans = load_dataset_metrics(args.haifa)
        print(f"Loaded Haifa Bay dataset: {len(haifa_rms)} segments.")
        has_haifa = True
    except Exception as e:
        print(f"Warning: Skipping Haifa Bay dataset: {e}")
        
    # Calculate statistics
    stats = {}
    datasets = [
        ("Scooter", scooter_rms, scooter_trans), 
        ("Croatia", croatia_rms, croatia_trans)
    ]
    if has_haifa:
        datasets.append(("Haifa", haifa_rms, haifa_trans))
    
    for name, rms, trans in datasets:
        # Ignore fallback silent values (-100 dBFS) for stats calculation if needed
        active_rms = rms[rms > -99.0] if np.any(rms > -99.0) else rms
        
        mean_rms = np.mean(active_rms) if len(active_rms) > 0 else -100.0
        median_rms = np.median(active_rms) if len(active_rms) > 0 else -100.0
        std_rms = np.std(active_rms) if len(active_rms) > 0 else 0.0
        max_rms = np.max(active_rms) if len(active_rms) > 0 else -100.0
        min_rms = np.min(active_rms) if len(active_rms) > 0 else -100.0
        
        mean_trans = np.mean(trans) if len(trans) > 0 else 0.0
        max_trans = np.max(trans) if len(trans) > 0 else 0.0
        
        # Count significant transitions (events)
        event_count = np.sum(trans >= args.threshold) if len(trans) > 0 else 0
        event_ratio = (event_count / len(trans)) * 100 if len(trans) > 0 else 0.0
        
        stats[name] = {
            "mean_rms": mean_rms,
            "median_rms": median_rms,
            "std_rms": std_rms,
            "max_rms": max_rms,
            "min_rms": min_rms,
            "mean_trans": mean_trans,
            "max_trans": max_trans,
            "event_count": event_count,
            "event_ratio": event_ratio
        }

    # Print comparison table
    print("\n" + "="*95)
    print("                      ACOUSTIC DATASETS COMPARISON REPORT")
    print("="*95)
    if has_haifa:
        print(f"| Metric                                | Scooter           | Croatia           | Haifa Bay         |")
        print(f"|---------------------------------------|-------------------|-------------------|-------------------|")
        print(f"| Mean RMS Loudness (dBFS)              | {stats['Scooter']['mean_rms']:17.2f} | {stats['Croatia']['mean_rms']:17.2f} | {stats['Haifa']['mean_rms']:17.2f} |")
        print(f"| Median RMS Loudness (dBFS)            | {stats['Scooter']['median_rms']:17.2f} | {stats['Croatia']['median_rms']:17.2f} | {stats['Haifa']['median_rms']:17.2f} |")
        print(f"| Loudness Variance (Std Dev)           | {stats['Scooter']['std_rms']:17.2f} | {stats['Croatia']['std_rms']:17.2f} | {stats['Haifa']['std_rms']:17.2f} |")
        print(f"| Max Segment Loudness (dBFS)           | {stats['Scooter']['max_rms']:17.2f} | {stats['Croatia']['max_rms']:17.2f} | {stats['Haifa']['max_rms']:17.2f} |")
        print(f"| Min Segment Loudness (dBFS)           | {stats['Scooter']['min_rms']:17.2f} | {stats['Croatia']['min_rms']:17.2f} | {stats['Haifa']['min_rms']:17.2f} |")
        print(f"| Mean Transition Step (dB)             | {stats['Scooter']['mean_trans']:17.2f} | {stats['Croatia']['mean_trans']:17.2f} | {stats['Haifa']['mean_trans']:17.2f} |")
        print(f"| Max Transition Step (dB)              | {stats['Scooter']['max_trans']:17.2f} | {stats['Croatia']['max_trans']:17.2f} | {stats['Haifa']['max_trans']:17.2f} |")
        print(f"| Event Density (steps >= {args.threshold} dB)    | {stats['Scooter']['event_ratio']:16.2f}% | {stats['Croatia']['event_ratio']:16.2f}% | {stats['Haifa']['event_ratio']:16.2f}% |")
    else:
        print(f"| Metric                                | Scooter           | Croatia           |")
        print(f"|---------------------------------------|-------------------|-------------------|")
        print(f"| Mean RMS Loudness (dBFS)              | {stats['Scooter']['mean_rms']:17.2f} | {stats['Croatia']['mean_rms']:17.2f} |")
        print(f"| Median RMS Loudness (dBFS)            | {stats['Scooter']['median_rms']:17.2f} | {stats['Croatia']['median_rms']:17.2f} |")
        print(f"| Loudness Variance (Std Dev)           | {stats['Scooter']['std_rms']:17.2f} | {stats['Croatia']['std_rms']:17.2f} |")
        print(f"| Max Segment Loudness (dBFS)           | {stats['Scooter']['max_rms']:17.2f} | {stats['Croatia']['max_rms']:17.2f} |")
        print(f"| Min Segment Loudness (dBFS)           | {stats['Scooter']['min_rms']:17.2f} | {stats['Croatia']['min_rms']:17.2f} |")
        print(f"| Mean Transition Step (dB)             | {stats['Scooter']['mean_trans']:17.2f} | {stats['Croatia']['mean_trans']:17.2f} |")
        print(f"| Max Transition Step (dB)              | {stats['Scooter']['max_trans']:17.2f} | {stats['Croatia']['max_trans']:17.2f} |")
        print(f"| Event Density (steps >= {args.threshold} dB)    | {stats['Scooter']['event_ratio']:16.2f}% | {stats['Croatia']['event_ratio']:16.2f}% |")
    print("="*95 + "\n")

    # Generate comparison visualization report
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    
    # Modern clean styling
    plt.rcParams["font.family"] = "sans-serif"
    colors = {
        "scooter": "#DD8452",  # Soft Orange/Coral
        "croatia": "#4C72B0",  # Soft Blue/Teal
        "haifa": "#55A868"     # Soft Green
    }
    
    title_str = "Comparative Acoustic Profile: Scooter vs Croatia"
    if has_haifa:
        title_str += " vs Haifa Bay"
    fig.suptitle(title_str, fontsize=16, fontweight="bold", y=0.96)
    
    # 1. Overlay Histograms
    axs[0, 0].hist(scooter_rms[scooter_rms > -99.0], bins=35, color=colors["scooter"], alpha=0.5, label="Scooter Dataset", edgecolor="black", density=True, histtype="stepfilled")
    axs[0, 0].hist(croatia_rms[croatia_rms > -99.0], bins=35, color=colors["croatia"], alpha=0.5, label="Croatia Dataset", edgecolor="black", density=True, histtype="stepfilled")
    if has_haifa:
        axs[0, 0].hist(haifa_rms[haifa_rms > -99.0], bins=35, color=colors["haifa"], alpha=0.5, label="Haifa Bay Dataset", edgecolor="black", density=True, histtype="stepfilled")
    axs[0, 0].set_title("Soundscape Loudness Probability Density (RMS dBFS)", fontsize=12, fontweight="bold")
    axs[0, 0].set_xlabel("Loudness (dBFS)", fontsize=10)
    axs[0, 0].set_ylabel("Probability Density", fontsize=10)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend()
    
    # 2. Side-by-side Boxplots of RMS
    box_data_rms = [scooter_rms[scooter_rms > -99.0], croatia_rms[croatia_rms > -99.0]]
    tick_labels = ["Scooter", "Croatia"]
    if has_haifa:
        box_data_rms.append(haifa_rms[haifa_rms > -99.0])
        tick_labels.append("Haifa")
        
    bp1 = axs[0, 1].boxplot(box_data_rms, tick_labels=tick_labels, patch_artist=True, widths=0.5)
    # Colors boxplots
    bp1["boxes"][0].set_facecolor(colors["scooter"])
    bp1["boxes"][0].set_alpha(0.7)
    bp1["boxes"][1].set_facecolor(colors["croatia"])
    bp1["boxes"][1].set_alpha(0.7)
    if has_haifa:
        bp1["boxes"][2].set_facecolor(colors["haifa"])
        bp1["boxes"][2].set_alpha(0.7)
    axs[0, 1].set_title("Loudness Spread & Median Comparison (RMS dBFS)", fontsize=12, fontweight="bold")
    axs[0, 1].set_ylabel("Loudness (dBFS)", fontsize=10)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # 3. Boxplots of Consecutive step transitions (Loudness change burstiness)
    box_data_trans = [scooter_trans, croatia_trans]
    if has_haifa:
        box_data_trans.append(haifa_trans)
        
    bp2 = axs[1, 0].boxplot(box_data_trans, tick_labels=tick_labels, patch_artist=True, widths=0.5)
    bp2["boxes"][0].set_facecolor(colors["scooter"])
    bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_facecolor(colors["croatia"])
    bp2["boxes"][1].set_alpha(0.7)
    if has_haifa:
        bp2["boxes"][2].set_facecolor(colors["haifa"])
        bp2["boxes"][2].set_alpha(0.7)
    axs[1, 0].set_title("Consecutive Loudness Step Transitions Spread", fontsize=12, fontweight="bold")
    axs[1, 0].set_ylabel("Step Change (dB)", fontsize=10)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # 4. Cumulative Distribution Function (CDF)
    # Scooter CDF
    sorted_scooter = np.sort(scooter_rms[scooter_rms > -99.0])
    y_scooter = np.arange(1, len(sorted_scooter) + 1) / len(sorted_scooter)
    axs[1, 1].plot(sorted_scooter, y_scooter, color=colors["scooter"], linewidth=2.5, label="Scooter Dataset")
    
    # Croatia CDF
    sorted_croatia = np.sort(croatia_rms[croatia_rms > -99.0])
    y_croatia = np.arange(1, len(sorted_croatia) + 1) / len(sorted_croatia)
    axs[1, 1].plot(sorted_croatia, y_croatia, color=colors["croatia"], linewidth=2.5, label="Croatia Dataset")
    
    # Haifa CDF
    if has_haifa:
        sorted_haifa = np.sort(haifa_rms[haifa_rms > -99.0])
        y_haifa = np.arange(1, len(sorted_haifa) + 1) / len(sorted_haifa)
        axs[1, 1].plot(sorted_haifa, y_haifa, color=colors["haifa"], linewidth=2.5, label="Haifa Bay Dataset")
    
    axs[1, 1].set_title("Loudness Cumulative Distribution Function (CDF)", fontsize=12, fontweight="bold")
    axs[1, 1].set_xlabel("Loudness (dBFS)", fontsize=10)
    axs[1, 1].set_ylabel("Fraction of Time Spent Below", fontsize=10)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save output plot
    plt.savefig(args.output_img, dpi=150)
    print(f"Comparison report plot saved to: {os.path.abspath(args.output_img)}")
    
    if args.extra_dir:
        os.makedirs(args.extra_dir, exist_ok=True)
        extra_copy = os.path.join(args.extra_dir, os.path.basename(args.output_img))
        plt.savefig(extra_copy, dpi=150)
        print(f"Extra copy of comparison plot saved to: {os.path.abspath(extra_copy)}")

if __name__ == "__main__":
    main()
