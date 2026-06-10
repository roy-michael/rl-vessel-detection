import os
import csv
import argparse
import re
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def parse_filename_datetime(filename):
    """
    Parses a YYYYMMDD_HHMMSS or YYYY-MM-DD_HH-MM-SS pattern from the filename.
    Example 1: RBW6737_20250725_080100.wav -> datetime(2025, 7, 25, 8, 1, 0)
    Example 2: channelA_2025-08-05_16-01-01.wav -> datetime(2025, 8, 5, 16, 1, 1)
    """
    match1 = re.search(r"(\d{8})_(\d{6})", filename)
    if match1:
        date_str, time_str = match1.groups()
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
            
    match2 = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
    if match2:
        date_str, time_str = match2.groups()
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
        except ValueError:
            pass
            
    return None

def load_data(csv_path, metric="euclidean", feature_type="rms"):
    """Loads CSV data and extracts rows, timestamps, and chosen feature vectors."""
    rows = []
    features = []
    timestamps = []
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
    use_db = metric.endswith("_db")
    fallback_base = datetime(2000, 1, 1, 0, 0, 0)
    elapsed_seconds = 0
    
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 1. Parse Amplitude Features
                if use_db:
                    peak_val = float(row["Peak_dBFS"])
                    rms_val = float(row["RMS_dBFS"])
                else:
                    peak_val = float(row["Peak_Amp"])
                    rms_val = float(row["RMS_Amp"])
                
                # Determine which feature to use
                if feature_type == "rms":
                    feat = [rms_val]
                elif feature_type == "peak":
                    feat = [peak_val]
                else:
                    feat = [peak_val, rms_val]
                
                # 2. Parse Datetime Timestamp
                file_dt = parse_filename_datetime(row["File"])
                start_offset = float(row["Start_Time"])
                
                if file_dt:
                    actual_dt = file_dt + timedelta(seconds=start_offset)
                else:
                    actual_dt = fallback_base + timedelta(seconds=elapsed_seconds)
                    elapsed_seconds += 1
                    
                # Format time portion only (HH:MM:SS)
                time_str = actual_dt.strftime("%H:%M:%S")
                row["Time"] = time_str
                
                rows.append(row)
                features.append(feat)
                timestamps.append(actual_dt)
                
            except (ValueError, KeyError):
                continue
                
    return rows, np.array(features), timestamps

def get_distance_func(metric):
    """Returns a distance function for single vector comparisons."""
    if metric in ("euclidean", "euclidean_db"):
        return lambda u, v: float(np.linalg.norm(u - v))
    elif metric in ("manhattan", "manhattan_db"):
        return lambda u, v: float(np.sum(np.abs(u - v)))
    elif metric == "cosine":
        return lambda u, v: float(1.0 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-10))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def calculate_consecutive_distances(rows, features, metric="euclidean"):
    """
    Calculates distance to the next segment using the selected metric.
    Resets at file boundaries.
    """
    dist_func = get_distance_func(metric)
    consecutive_dist = []
    n = len(rows)
    
    for i in range(n):
        current_row = rows[i]
        current_feat = features[i]
        
        if i == n - 1 or rows[i+1]["File"] != current_row["File"]:
            dist = 0.0
        else:
            next_feat = features[i+1]
            dist = dist_func(current_feat, next_feat)
            
        consecutive_dist.append(dist)
        current_row["Distance_to_Next"] = round(dist, 5)
        
    return np.array(consecutive_dist)

def calculate_pairwise_distances(features, metric="euclidean"):
    """Computes the pairwise distance matrix for the selected metric."""
    n = len(features)
    
    if metric in ("euclidean", "euclidean_db"):
        sq_norms = np.sum(features**2, axis=1)
        dist_matrix = np.sqrt(np.maximum(
            sq_norms[:, np.newaxis] + sq_norms[np.newaxis] - 2 * np.dot(features, features.T), 
            0.0
        ))
        
    elif metric in ("manhattan", "manhattan_db"):
        # Vectorized absolute difference sum along feature dimension
        dist_matrix = np.sum(np.abs(features[:, np.newaxis, :] - features[np.newaxis, :, :]), axis=-1)
                      
    elif metric == "cosine":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed_features = features / norms
        dist_matrix = 1.0 - np.dot(normed_features, normed_features.T)
        
    else:
        raise ValueError(f"Unknown metric: {metric}")
        
    return dist_matrix

def main():
    parser = argparse.ArgumentParser(description="Calculate distance between audio segment features using time timeline.")
    parser.add_argument("--csv", default="scooter_amplitude_analysis.csv", help="Input CSV file")
    parser.add_argument("--output-csv", default="scooter_segment_distances.csv", help="Output CSV file")
    parser.add_argument("--consecutive-plot", default=None, help="Consecutive plot filename")
    parser.add_argument("--pairwise-plot", default=None, help="Heatmap filename")
    parser.add_argument("--metric", default="manhattan_db", choices=["euclidean", "manhattan", "cosine", "euclidean_db", "manhattan_db"],
                        help="Distance metric to use (default: manhattan_db)")
    parser.add_argument("--feature", default="rms", choices=["rms", "peak", "both"],
                        help="Amplitude feature to use: 'rms', 'peak', or 'both' (default: rms)")
    parser.add_argument("--extra-dir", default=None, help="Directory to save extra plot copies")
    
    args = parser.parse_args()
    
    print(f"Using feature: {args.feature.upper()} | metric: {args.metric.upper()}")
    
    try:
        rows, features, timestamps = load_data(args.csv, args.metric, args.feature)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    if len(rows) == 0:
        print("No valid data found.")
        return
        
    print(f"Loaded {len(rows)} segment features.")
    
    # 1. Calculate consecutive distances
    consecutive_dists = calculate_consecutive_distances(rows, features, args.metric)
    
    # 2. Find top 10 segments with largest consecutive transitions
    largest_transition_idx = np.argsort(consecutive_dists)[::-1]
    print(f"\nTop 10 Largest Consecutive Transitions ({args.metric.upper()}):")
    print("  Rank | Time     | File                           | Seg | Peak / RMS   | Distance to Next")
    print("  -----+----------+--------------------------------+-----+--------------+------------------")
    rank = 1
    for idx in largest_transition_idx:
        row = rows[idx]
        if float(row["Distance_to_Next"]) == 0.0:
            continue
        feat_str = f"{float(row['Peak_Amp']):.4f} / {float(row['RMS_Amp']):.4f}"
        print(f"   {rank:2d}  | {row['Time']} | {row['File'][:30]:30s} | {int(row['Segment']):3d} | {feat_str:12s} | {float(row['Distance_to_Next']):.5f}")
        rank += 1
        if rank > 10:
            break
            
    # 3. Find top 10 most similar distinct pairs (excluding self-similar pairs)
    n = len(rows)
    if features.shape[1] == 1:
        # Optimized 1D similarity finder (O(N log N) time, O(N) memory) - Prevents memory crash on large datasets
        flat_feats = features.flatten()
        sorted_idx = np.argsort(flat_feats)
        adj_diffs = np.diff(flat_feats[sorted_idx])
        sorted_diff_idx = np.argsort(adj_diffs)
        
        print(f"\nTop 10 Most Similar Segment Pairs ({args.metric.upper()}):")
        print("  Rank | Time A   (Seg A)             | Time B   (Seg B)             | Dist | Feat A [Peak, RMS] | Feat B [Peak, RMS]")
        print("  -----+------------------------------+------------------------------+------+--------------------+--------------------")
        
        rank = 1
        seen_pairs = set()
        for idx in sorted_diff_idx:
            i = sorted_idx[idx]
            j = sorted_idx[idx + 1]
            
            pair = tuple(sorted([i, j]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            row_i = rows[i]
            row_j = rows[j]
            dist_val = adj_diffs[idx]
            
            feat_i_str = f"[{float(row_i['Peak_Amp']):.3f}, {float(row_i['RMS_Amp']):.3f}]"
            feat_j_str = f"[{float(row_j['Peak_Amp']):.3f}, {float(row_j['RMS_Amp']):.3f}]"
            
            label_i = f"{row_i['Time']} (S{row_i['Segment']})"
            label_j = f"{row_j['Time']} (S{row_j['Segment']})"
            
            print(f"   {rank:2d}  | {label_i:28s} | {label_j:28s} | {dist_val:.5f} | {feat_i_str:18s} | {feat_j_str:18s}")
            
            rank += 1
            if rank > 10:
                break
    else:
        # Standard 2D similarity search - only if dataset is small
        if n > 3000:
            print("\nWarning: Dataset is too large to find global similar pairs in 2D space. Skipping similarity search.")
        else:
            dist_matrix = calculate_pairwise_distances(features, args.metric)
            masked_matrix = dist_matrix.copy()
            np.fill_diagonal(masked_matrix, np.inf)
            
            triu_indices = np.triu_indices(n, k=1)
            triu_distances = masked_matrix[triu_indices]
            sorted_triu_idx = np.argsort(triu_distances)
            
            print(f"\nTop 10 Most Similar Segment Pairs ({args.metric.upper()}):")
            print("  Rank | Time A   (Seg A)             | Time B   (Seg B)             | Dist | Feat A [Peak, RMS] | Feat B [Peak, RMS]")
            print("  -----+------------------------------+------------------------------+------+--------------------+--------------------")
            for rank in range(1, 11):
                idx = sorted_triu_idx[rank - 1]
                i = triu_indices[0][idx]
                j = triu_indices[1][idx]
                
                row_i = rows[i]
                row_j = rows[j]
                
                feat_i_str = f"[{float(row_i['Peak_Amp']):.3f}, {float(row_i['RMS_Amp']):.3f}]"
                feat_j_str = f"[{float(row_j['Peak_Amp']):.3f}, {float(row_j['RMS_Amp']):.3f}]"
                
                label_i = f"{row_i['Time']} (S{row_i['Segment']})"
                label_j = f"{row_j['Time']} (S{row_j['Segment']})"
                
                print(f"   {rank:2d}  | {label_i:28s} | {label_j:28s} | {triu_distances[idx]:.5f} | {feat_i_str:18s} | {feat_j_str:18s}")
        
    # 4. Save results to a new CSV file
    original_keys = list(rows[0].keys())
    for key_to_remove in ('Timestamp', 'Time', 'File'):
        if key_to_remove in original_keys:
            original_keys.remove(key_to_remove)
            
    fieldnames = ['File', 'Time'] + original_keys
    
    try:
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nTime timeline distance data saved to: {os.path.abspath(args.output_csv)}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")
        
    # 5. Plot consecutive distance timeline using times (HH:MM:SS)
    if args.consecutive_plot:
        plt.figure(figsize=(13, 6))
        plt.plot(timestamps, consecutive_dists, color="teal", alpha=0.8, linewidth=1.5, label=f"Distance ({args.metric})")
        plt.title(f"Segment Consecutive Transition Distance ({args.metric.upper()}) over Recording Time Timeline\n(Feature: {args.feature.upper()})", fontsize=13, fontweight="bold")
        plt.xlabel("Recording Time (HH:MM:SS)", fontsize=11)
        plt.ylabel(f"Distance ({args.metric.upper()})", fontsize=11)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.consecutive_plot, dpi=150)
        print(f"Consecutive time timeline plot saved to: {os.path.abspath(args.consecutive_plot)}")
        
    # 6. Plot pairwise similarity heatmap (using safety downsampling for large datasets)
    if args.pairwise_plot:
        max_pairwise_n = 2000
        if n > max_pairwise_n:
            print(f"\nWarning: Dataset size ({n} segments) is too large for full pairwise distance matrix.")
            print(f"Downsampling to {max_pairwise_n} segments for pairwise similarity heatmap...")
            step = int(np.ceil(n / max_pairwise_n))
            pairwise_features = features[::step]
            dist_matrix = calculate_pairwise_distances(pairwise_features, args.metric)
            heatmap_title = f"Pairwise Distance Heatmap ({args.metric.upper()})\n(Downsampled: Every {step}th Segment)"
        else:
            dist_matrix = calculate_pairwise_distances(features, args.metric)
            heatmap_title = f"Pairwise Distance Heatmap ({args.metric.upper()})\n(Feature: {args.feature.upper()})"

        plt.figure(figsize=(10, 8))
        plt.imshow(dist_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label=f"Distance ({args.metric.upper()}) - lower is more similar")
        plt.title(heatmap_title, fontsize=13, fontweight="bold")
        plt.xlabel("Segment Chronological Index", fontsize=11)
        plt.ylabel("Segment Chronological Index", fontsize=11)
        plt.tight_layout()
        plt.savefig(args.pairwise_plot, dpi=150)
        print(f"Pairwise heatmap saved to: {os.path.abspath(args.pairwise_plot)}")
        
    # Save extra copies if extra dir is specified
    if args.extra_dir:
        os.makedirs(args.extra_dir, exist_ok=True)
        if args.consecutive_plot:
            extra_consecutive = os.path.join(args.extra_dir, os.path.basename(args.consecutive_plot))
            plt.figure(1)
            plt.savefig(extra_consecutive, dpi=150)
        if args.pairwise_plot:
            extra_pairwise = os.path.join(args.extra_dir, os.path.basename(args.pairwise_plot))
            plt.figure(2)
            plt.savefig(extra_pairwise, dpi=150)
        print(f"Extra copies saved to: {os.path.abspath(args.extra_dir)}")

if __name__ == "__main__":
    main()
