import os
import asyncio

import librosa
import matplotlib.pyplot as plt
import numpy as np

from core.agent import DispatcherAgent, SignalProcessorAgent
from core.environment import Environment

# Override by setting the RECORDINGS_DIR environment variable
BASE_DIR = os.environ.get("RECORDINGS_DIR", "c:/Users/Roy/Recordings")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Vessel Tracker")
    parser.add_argument("--dataset", type=str, choices=["croatia", "scooter"], default="croatia", help="Dataset to run on")
    args = parser.parse_args()

    if args.dataset == "scooter":
        file_dir = f"{BASE_DIR}/scooter"
        report_filename = "scooter_vessel_detection_report.txt"
        img_filename = "scooter_vessel_detection_timeline.png"
    else:
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2507_1"
        report_filename = "vessel_detection_report.txt"
        img_filename = "vessel_detection_timeline.png"

    min_freq = 40
    max_freq = 2000
    n_fft = 16 * 1024
    hop_length = n_fft // 2 # int(1 * 1024)
    window_sec = 15.0
    n_components = 8

    # Parameters for component clustering and variance scaling
    proximity_threshold_hz = 25.0
    association_threshold_hz = 30.0
    peak_spread_window_bins = 15
    variance_multiplier = 1.5
    min_variance_floor = 25.0
    consolidation_threshold_hz = 25.0

    try:
        env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length)
        agent = DispatcherAgent(env, min_freq, max_freq, n_fft, n_components, 2000, window_sec)
        signal_processor = SignalProcessorAgent(
            agent,
            proximity_threshold_hz=proximity_threshold_hz,
            association_threshold_hz=association_threshold_hz,
            peak_spread_window_bins=peak_spread_window_bins,
            variance_multiplier=variance_multiplier,
            min_variance_floor=min_variance_floor,
            consolidation_threshold_hz=consolidation_threshold_hz
        )

    except ValueError as e:
        print(e)
        return

    await env.start()
    
    # Wait for the environment to buffer the first frame so we have something to plot immediately
    first_frame, _ = await env.peek()
    if first_frame is None:
         print("No data available to start.")
         return
         
    # Start the agent, which will now handle the plotting and observation loop
    await agent.start()
    await signal_processor.start()
    
    # Keep the main thread alive while the agent runs
    # In a real UI application, this would be replaced by the UI event loop
    try:
        while not agent.completed:
            await asyncio.sleep(0.5)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        report_lines = []
        def log(msg=""):
            print(msg)
            report_lines.append(msg)

        log("\n" + "="*70)
        log("                 FINAL VESSEL DETECTION & SPEED REPORT")
        log("="*70)
        states = list(signal_processor.tracker.states)
        active_states = [
            s for s in signal_processor.tracker.active_states.values()
            if (agent.current_time - s.start_time) >= signal_processor.tracker.min_duration_sec
        ]
        all_states = states + active_states
            
        if not all_states:
            log("No stable vessel states tracked during this run.")
        else:
            # Group states by Vessel ID
            vessel_history = {}
            for state in all_states:
                vid = state.vessel_id if state.vessel_id else "Noise"
                if vid not in vessel_history:
                    vessel_history[vid] = []
                vessel_history[vid].append(state)
                
            # Filter and sort Vessel IDs by cumulative duration (most active first)
            vessel_durations = []
            for vid, segs in vessel_history.items():
                total_duration = sum((s.end_time or agent.current_time) - s.start_time for s in segs)
                vessel_durations.append((vid, total_duration, segs))
                
            # Sort so dominant vessels are printed at the top
            vessel_durations.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (vid, dur, segs) in enumerate(vessel_durations):
                is_dominant = (rank < 8)  # top 8 are dominant
                dominance_status = "DOMINANT TARGET" if is_dominant else "BACKGROUND NOISE/TRANSIENT"
                
                # Sort segments chronologically
                segs_sorted = sorted(segs, key=lambda s: s.start_time)
                
                # Compute absolute time frame and overall weighted mean amplitude
                min_start = min(s.start_time for s in segs_sorted)
                max_end = max((s.end_time or agent.current_time) for s in segs_sorted)
                
                total_duration = sum((s.end_time or agent.current_time) - s.start_time for s in segs_sorted)
                if total_duration > 0:
                    overall_amp = sum(np.mean(s.amplitudes) * ((s.end_time or agent.current_time) - s.start_time) for s in segs_sorted) / total_duration
                else:
                    overall_amp = 0.0
                
                log(f"\n>>> {vid} [{dominance_status}] (Total Active Duration: {dur:.1f}s, Weighted Mean Amp: {overall_amp:.4f}):")
                log(f"  Absolute Active Time Window: {min_start:.1f}s - {max_end:.1f}s")
                for s_idx, state in enumerate(segs_sorted):
                    status = "ACTIVE" if state in active_states else "COMPLETED"
                    s_dur = (state.end_time or agent.current_time) - state.start_time
                    mean_amp = np.mean(state.amplitudes) if state.amplitudes else 0.0
                    log(f"  Speed Stage {s_idx+1} [{status}]:")
                    log(f"    Interval:       {state.start_time:.1f}s - {(state.end_time or agent.current_time):.1f}s (Duration: {s_dur:.1f}s)")
                    log(f"    Mean Frequency: {state.mean_frequency:.1f} Hz")
                    log(f"    Mean Amplitude: {mean_amp:.4f}")
                    log(f"    Total Variance: {state.total_variance:.1f} Hz^2 (Std Dev: {np.sqrt(state.total_variance):.1f} Hz)")
                log("-" * 55)
        log("="*70 + "\n")

        # Save the textual report to a file
        try:
            output_report_path = os.path.join("output", report_filename)
            with open(output_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Saved textual report to {output_report_path}")
        except Exception as e:
            print(f"Could not save textual report: {e}")

        # Force a final tracker plot update to capture all observations
        if signal_processor and agent and hasattr(agent, 'fig') and plt.fignum_exists(agent.fig.number):
            signal_processor._update_annotations()
            try:
                signal_processor._update_tracker_plot()
            except Exception as e:
                print(f"Could not perform final plot update: {e}")

        # Save the matplotlib figure to an image
        if agent and hasattr(agent, 'fig') and plt.fignum_exists(agent.fig.number):
            try:
                output_img_path = os.path.join("output", img_filename)
                agent.fig.savefig(output_img_path, bbox_inches="tight", dpi=150)
                print(f"Saved final graph to {output_img_path}")
            except Exception as e:
                print(f"Could not save figure image: {e}")

        # Generate and save a simulated spectrogram of only the tracked signals
        if all_states:
            try:
                print("Generating simulated spectrogram...")
                max_time = agent.current_time
                dt = 0.5
                t_grid = np.arange(0, max_time, dt)
                f_grid = np.linspace(min_freq, max_freq, 800)
                spec_grid = np.zeros((len(f_grid), len(t_grid)))
                
                for state in all_states:
                    start_t = state.start_time
                    end_t = state.end_time or max_time
                    
                    # Frequency, amplitude, and spread arrays for the state
                    freqs_arr = np.array(state.frequencies)
                    amps_arr = np.array(state.amplitudes)
                    spreads_arr = np.array(state.spreads) if hasattr(state, 'spreads') else np.ones_like(freqs_arr) * 10.0
                    
                    if len(freqs_arr) == 0:
                        continue
                        
                    # Interpolate values over the segment's active duration on the time grid
                    seg_t_mask = (t_grid >= start_t) & (t_grid <= end_t)
                    seg_ts = t_grid[seg_t_mask]
                    if len(seg_ts) == 0:
                        continue
                        
                    # Linear interpolation mapping segment duration to index space
                    rel_indices = np.linspace(0, len(freqs_arr) - 1, len(seg_ts))
                    interp_freqs = np.interp(rel_indices, np.arange(len(freqs_arr)), freqs_arr)
                    interp_amps = np.interp(rel_indices, np.arange(len(amps_arr)), amps_arr)
                    interp_spreads = np.interp(rel_indices, np.arange(len(spreads_arr)), spreads_arr)
                    
                    # Add Gaussians to the spectrogram grid
                    for col_idx, t_val in enumerate(t_grid):
                        if not seg_t_mask[col_idx]:
                            continue
                        # Find the corresponding interpolated values
                        seg_ts_idx = np.where(seg_ts == t_val)[0][0]
                        f_t = interp_freqs[seg_ts_idx]
                        amp_t = interp_amps[seg_ts_idx]
                        spread_t = max(5.0, interp_spreads[seg_ts_idx])
                        
                        # Add Gaussian peak to the frequency column
                        gaussian_profile = np.exp(-0.5 * ((f_grid - f_t) / spread_t) ** 2)
                        spec_grid[:, col_idx] += amp_t * gaussian_profile
                
                # Plot simulated spectrogram
                fig_sim, ax_sim = plt.subplots(figsize=(12, 6))
                # dB conversion of simulated activations for visual dynamic range
                spec_grid_db = 20 * np.log10(spec_grid + 1e-4)
                
                im_sim = ax_sim.imshow(
                    spec_grid_db,
                    aspect="auto",
                    origin="lower",
                    extent=[0, max_time, min_freq, max_freq],
                    cmap="inferno",
                    vmin=-60,
                    vmax=np.max(spec_grid_db)
                )
                fig_sim.colorbar(im_sim, ax=ax_sim, label="Simulated Amplitude (dB)")
                ax_sim.set_xlabel("Absolute Time (seconds)", fontsize=11, fontweight="bold")
                ax_sim.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
                ax_sim.set_title(f"Simulated Spectrogram (Tracked Vessel Signals Only) - {args.dataset.upper()}", fontsize=13, fontweight="bold")
                ax_sim.grid(True, linestyle="--", alpha=0.3)
                
                sim_spectrogram_filename = f"{args.dataset}_simulated_spectrogram.png"
                sim_spectrogram_path = os.path.join("output", sim_spectrogram_filename)
                fig_sim.savefig(sim_spectrogram_path, bbox_inches="tight", dpi=150)
                plt.close(fig_sim)
                print(f"Saved simulated spectrogram to {sim_spectrogram_path}")
            except Exception as e:
                print(f"Could not generate simulated spectrogram: {e}")

if __name__ == "__main__":
    asyncio.run(main())