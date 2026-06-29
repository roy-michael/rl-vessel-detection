import os
import glob
import argparse
import numpy as np
import librosa
import csv

def safe_db(value, min_db=-100.0):
    """
    Safely converts a linear amplitude value to dBFS.
    Avoids log10(0) by capping values below 1e-10 (which is -200 dB).
    """
    if value <= 0:
        return min_db
    db = 20 * np.log10(value)
    return max(db, min_db)

def analyze_audio_file(filepath, segment_len_sec=1.0):
    """
    Loads an audio file, splits it into segment_len_sec chunks,
    and analyzes peak and RMS amplitude metrics for each segment.
    """
    filename = os.path.basename(filepath)
    print(f"\nAnalyzing: {filename}")
    
    try:
        # Load audio file (mono=True to combine channels for simplified amplitude analysis)
        y, sr = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

    duration = librosa.get_duration(y=y, sr=sr)
    total_samples = len(y)
    samples_per_segment = int(sr * segment_len_sec)
    
    print(f"  Sample Rate: {sr} Hz")
    # Using format specifier for duration and total_samples
    print(f"  Duration: {duration:.2f} seconds ({total_samples} samples)")

    results = []
    
    # Process audio in 1-second steps
    for i, start_sample in enumerate(range(0, total_samples, samples_per_segment)):
        end_sample = min(start_sample + samples_per_segment, total_samples)
        segment = y[start_sample:end_sample]
        
        # Calculate actual segment duration (might be less than 1.0s for the last segment)
        seg_duration = (end_sample - start_sample) / sr
        
        # Calculate absolute peak amplitude
        peak_amp = float(np.max(np.abs(segment))) if len(segment) > 0 else 0.0
        peak_db = safe_db(peak_amp)
        
        # Calculate RMS (Root Mean Square) amplitude
        rms_amp = float(np.sqrt(np.mean(segment**2))) if len(segment) > 0 else 0.0
        rms_db = safe_db(rms_amp)
        
        start_time = start_sample / sr
        end_time = end_sample / sr
        segment_num = i + 1
        
        results.append({
            "File": filename,
            "Segment": segment_num,
            "Start_Time": round(start_time, 3),
            "End_Time": round(end_time, 3),
            "Duration_Sec": round(seg_duration, 3),
            "Peak_Amp": round(peak_amp, 5),
            "Peak_dBFS": round(peak_db, 2),
            "RMS_Amp": round(rms_amp, 5),
            "RMS_dBFS": round(rms_db, 2)
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze audio signal amplitudes in 1-second segments.")
    parser.add_argument("--dir", default=r"C:\Users\Roy\Recordings\scooter", help="Directory containing audio files to analyze (default: scooter recordings directory)")
    parser.add_argument("--output", default="scooter_amplitude_analysis.csv", help="Path to output CSV file (default: scooter_amplitude_analysis.csv)")
    parser.add_argument("--segment-len", type=float, default=1.0, help="Segment duration in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    # Supported audio file extensions
    extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
    
    audio_files = []
    for ext in extensions:
        # Search both lower and upper case extensions
        audio_files.extend(glob.glob(os.path.join(args.dir, ext)))
        audio_files.extend(glob.glob(os.path.join(args.dir, ext.upper())))
    
    # Remove duplicates if any (due to case-insensitivity on Windows or double matches)
    audio_files = sorted(list(set(os.path.abspath(f) for f in audio_files)))
    
    if not audio_files:
        print(f"No audio files found in directory: {os.path.abspath(args.dir)}")
        print("Supported formats: WAV, MP3, FLAC, OGG, M4A")
        return
        
    print(f"Found {len(audio_files)} audio file(s) to analyze.")
    
    all_results = []
    for filepath in audio_files:
        results = analyze_audio_file(filepath, args.segment_len)
        all_results.extend(results)
        
        # Display a summary table for this file in the console
        if results:
            print("\n  Segment | Start (s) | End (s) | Duration | Peak Amp | Peak (dBFS) | RMS Amp  | RMS (dBFS)")
            print("  --------+-----------+---------+----------+----------+-------------+----------+-----------")
            for res in results:
                print(f"  {res['Segment']:7d} | {res['Start_Time']:9.2f} | {res['End_Time']:7.2f} | {res['Duration_Sec']:7.2f}s | {res['Peak_Amp']:8.4f} | {res['Peak_dBFS']:11.2f} | {res['RMS_Amp']:8.4f} | {res['RMS_dBFS']:9.2f}")
            print()

    if all_results:
        # Write to CSV
        fieldnames = ["File", "Segment", "Start_Time", "End_Time", "Duration_Sec", "Peak_Amp", "Peak_dBFS", "RMS_Amp", "RMS_dBFS"]
        try:
            with open(args.output, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            print(f"Analysis results saved to: {os.path.abspath(args.output)}")
        except Exception as e:
            print(f"Failed to write results to CSV file: {e}")

if __name__ == "__main__":
    main()
