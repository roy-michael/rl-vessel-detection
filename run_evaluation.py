import os
import asyncio

import librosa
import matplotlib.pyplot as plt
import numpy as np

from core.agent import DSPOrchestrator, VesselTrackProcessor
from core.environment import AcousticDataStreamer, TrackingMDPEnv

# Override by setting the RECORDINGS_DIR environment variable
BASE_DIR = os.environ.get("RECORDINGS_DIR", "C:/Users/Roy/Recordings")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Vessel Tracker")
    parser.add_argument("--dataset", type=str,
                        choices=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"],
                        default="croatia", help="Dataset to run on")
    parser.add_argument("--rl-agent", type=str,
                        choices=["double_q_learning", "linear_fa", "actor_critic"],
                        default="double_q_learning", help="RL agent policy to use")
    parser.add_argument("--policy-dataset", type=str,
                        choices=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"],
                        default="croatia", help="Dataset the policy was trained on")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display")
    args = parser.parse_args()

    _ds_prefix = {
        "croatia":         "croatia_2507_1",
        "croatia_2507_2":  "croatia_2507_2",
        "croatia_2407_1":  "croatia_2407_1",
        "croatia_2407_2":  "croatia_2407_2",
        "croatia_2307":    "croatia_2307",
        "scooter":         "scooter",
    }.get(args.dataset, args.dataset)
    
    out_dir = os.path.join("output", _ds_prefix)
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == "scooter":
        file_dir = f"{BASE_DIR}/DepartmentalCruise-2025-06-12/icListen/wav"
        report_filename = f"scooter_{args.rl_agent}_vessel_detection_report.txt"
        img_filename = f"scooter_{args.rl_agent}_vessel_detection_timeline.png"
        min_freq = 400
    elif args.dataset == "croatia_2507_2":
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2507_2_joint"
        report_filename = f"croatia_2507_2_{args.rl_agent}_report.txt"
        img_filename = f"croatia_2507_2_{args.rl_agent}_timeline.png"
        min_freq = 40
    elif args.dataset == "croatia_2407_1":
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2407_1_600m"
        report_filename = f"croatia_2407_1_{args.rl_agent}_report.txt"
        img_filename = f"croatia_2407_1_{args.rl_agent}_timeline.png"
        min_freq = 40
    elif args.dataset == "croatia_2407_2":
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2407_2_snake"
        report_filename = f"croatia_2407_2_{args.rl_agent}_report.txt"
        img_filename = f"croatia_2407_2_{args.rl_agent}_timeline.png"
        min_freq = 400
    elif args.dataset == "croatia_2307":
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2307_free"
        report_filename = f"croatia_2307_{args.rl_agent}_report.txt"
        img_filename = f"croatia_2307_{args.rl_agent}_timeline.png"
        min_freq = 400
    else:
        croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
        file_dir = f"{croatia_base_dir}/2507_1_1k"
        report_filename = f"croatia_2507_1_{args.rl_agent}_report.txt"
        img_filename = f"croatia_2507_1_{args.rl_agent}_timeline.png"
        min_freq = 40
    max_freq = 4000
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
        env = AcousticDataStreamer(file_dir, min_freq, max_freq, n_fft, hop_length, max_files=args.max_files)
        agent = DSPOrchestrator(
            env, min_freq, max_freq, n_fft, n_components, 2000, window_sec, headless=args.headless,
            proximity_threshold_hz=proximity_threshold_hz,
            association_threshold_hz=association_threshold_hz,
            peak_spread_window_bins=peak_spread_window_bins,
            variance_multiplier=variance_multiplier,
            min_variance_floor=min_variance_floor,
            consolidation_threshold_hz=consolidation_threshold_hz,
            min_duration_sec=45.0,
            min_vessel_score=0.50
        )
        agent.dataset_name = args.dataset
        agent.policy_name = args.rl_agent
        signal_processor = agent
        
        # Load and integrate trained RL policy
        from core.agent.rl_agent import RLAgent
        from core.agent.policy import DoubleQLearningPolicy, LinearFAPolicy, ActorCriticPolicy
        
        _linear_fa = (args.rl_agent == "linear_fa")
        policy_dir = {
            "croatia":        "croatia_2507_1",
            "croatia_2507_2": "croatia_2507_2",
            "croatia_2407_1": "croatia_2407_1",
            "croatia_2407_2": "croatia_2407_2",
            "croatia_2307":   "croatia_2307",
            "scooter":        "scooter",
        }.get(args.policy_dataset, args.policy_dataset)
        policy_path = (
            f"models/{policy_dir}/rl_{args.rl_agent}_{args.policy_dataset}.npy" if _linear_fa
            else f"models/{policy_dir}/rl_{args.rl_agent}_{args.policy_dataset}.json"
        )
        if os.path.exists(policy_path):
            if args.rl_agent == "double_q_learning":
                policy = DoubleQLearningPolicy()
            elif args.rl_agent == "linear_fa":
                policy = LinearFAPolicy()
            elif args.rl_agent == "actor_critic":
                policy = ActorCriticPolicy()
            else:
                raise ValueError(f"Unknown rl_agent: {args.rl_agent}")
            
            rl_agent = RLAgent(policy=policy, epsilon=0.0)
            rl_agent.load_policy(policy_path)
            rl_env = TrackingMDPEnv(signal_processor.tracker)
            
            signal_processor.tracker.rl_agent = rl_agent
            signal_processor.tracker.rl_env = rl_env
            signal_processor.tracker.rl_epsilon = 0.0
            signal_processor.tracker.rl_stats = {
                'total_reward': 0.0,
                'action_counts': {0: 0, 1: 0, 2: 0},
                'status_counts': {}
            }
            print(f"Successfully integrated trained {args.rl_agent} tracking policy (trained on {args.policy_dataset}).")
        else:
            print(f"Warning: Trained {args.rl_agent} policy (trained on {args.policy_dataset}) not found at {policy_path}. Falling back to heuristic tracking rules.")

    except ValueError as e:
        print(e)
        return

    # Collect raw detections for the joint histogram
    raw_detections = []
    original_update_multi = signal_processor.tracker.update_multi
    def custom_update_multi(current_time, detections):
        for det in detections:
            if det['centroid'] > 0:
                raw_detections.append((det['centroid'], det['amplitude']))
        return original_update_multi(current_time, detections)
    signal_processor.tracker.update_multi = custom_update_multi

    await env.start()
    
    # Wait for the environment to buffer the first frame so we have something to plot immediately
    first_frame, _ = await env.peek()
    if first_frame is None:
         print("No data available to start.")
         return
         
    # Start the agent, which will now handle the plotting and observation loop
    await agent.start()
    
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
        
        # Run final consolidation to merge any remaining states at the end of the run
        signal_processor.tracker.consolidate_all_vessels()
        
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
                if total_duration >= 180.0:
                    vessel_durations.append((vid, total_duration, segs))
                
            # Sort so dominant vessels are printed at the top
            vessel_durations.sort(key=lambda x: x[1], reverse=True)
            
            # Print a high-level summary table at the top of the report to make all vessels visible
            log("\n" + "="*70)
            log("                 HIGH-LEVEL DOMINANT TARGETS SUMMARY")
            log("="*70)
            log(f"{'Vessel ID':<15} | {'Type/Status':<16} | {'Active Time Window':<19} | {'Total Duration':<14} | {'Mean Freq (Hz)':<14}")
            log("-" * 88)
            for rank, (vid, dur, segs) in enumerate(vessel_durations):
                is_dominant = (rank < 8)
                status = "DOMINANT TARGET" if is_dominant else "BACKGROUND NOISE"
                segs_sorted = sorted(segs, key=lambda s: s.start_time)
                min_start = min(s.start_time for s in segs_sorted)
                max_end = max((s.end_time or agent.current_time) for s in segs_sorted)
                mean_freq = np.mean([s.mean_frequency for s in segs_sorted])
                import datetime
                h_start = datetime.datetime.fromtimestamp(min_start).strftime("%H:%M:%S")
                h_end = datetime.datetime.fromtimestamp(max_end).strftime("%H:%M:%S")
                log(f"{vid:<15} | {status:<16} | {h_start} - {h_end} ({min_start:.1f}s - {max_end:.1f}s) | {dur:.1f}s | {mean_freq:.1f} Hz")
            log("="*70 + "\n")
            
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
                
                import datetime
                h_start = datetime.datetime.fromtimestamp(min_start).strftime("%H:%M:%S")
                h_end = datetime.datetime.fromtimestamp(max_end).strftime("%H:%M:%S")
                log(f"\n>>> {vid} [{dominance_status}] (Total Active Duration: {dur:.1f}s, Weighted Mean Amp: {overall_amp:.4f}):")
                log(f"  Absolute Active Time Window: {min_start:.1f}s - {max_end:.1f}s ({h_start} - {h_end})")
                for s_idx, state in enumerate(segs_sorted):
                    status = "ACTIVE" if state in active_states else "COMPLETED"
                    s_dur = (state.end_time or agent.current_time) - state.start_time
                    mean_amp = np.mean(state.amplitudes) if state.amplitudes else 0.0
                    h_s_start = datetime.datetime.fromtimestamp(state.start_time).strftime("%H:%M:%S")
                    h_s_end = datetime.datetime.fromtimestamp(state.end_time or agent.current_time).strftime("%H:%M:%S")
                    log(f"  Speed Stage {s_idx+1} [{status}]:")
                    log(f"    Interval:       {state.start_time:.1f}s - {(state.end_time or agent.current_time):.1f}s ({h_s_start} - {h_s_end}) (Duration: {s_dur:.1f}s)")
                    log(f"    Mean Frequency: {state.mean_frequency:.1f} Hz")
                    log(f"    Mean Amplitude: {mean_amp:.4f}")
                    log(f"    Total Variance: {state.total_variance:.1f} Hz^2 (Std Dev: {np.sqrt(state.total_variance):.1f} Hz)")
                log("-" * 55)
        log("="*70 + "\n")

        # If running in RL mode, log the RL stats
        if getattr(signal_processor.tracker, 'rl_stats', None) is not None:
            rl_stats = signal_processor.tracker.rl_stats
            log("="*70)
            log("                 REINFORCEMENT LEARNING EVALUATION METRICS")
            log("="*70)
            log(f"  RL Policy:           {args.rl_agent} (trained on {args.policy_dataset})")
            log(f"  Cumulative Reward:   {rl_stats['total_reward']:.1f}")
            action_counts = rl_stats['action_counts']
            log(f"  Actions Taken:       Reject (A0): {action_counts.get(0, 0)}, Associate (A1): {action_counts.get(1, 0)}, Spawn (A2): {action_counts.get(2, 0)}")
            log(f"  Outcome Statuses:    {dict(rl_stats['status_counts'])}")
            log("="*70 + "\n")

        # Save the textual report to a file
        try:
            output_report_path = os.path.join(out_dir, report_filename)
            with open(output_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Saved textual report to {output_report_path}")
        except Exception as e:
            print(f"Could not save textual report: {e}")

        # Force a final tracker plot update or recreate it offline for headless runs
        try:
            output_img_path = os.path.join(out_dir, img_filename)
            if agent and hasattr(agent, 'fig') and plt.fignum_exists(agent.fig.number):
                signal_processor._update_annotations()
                signal_processor._update_tracker_plot()
                agent.fig.savefig(output_img_path, bbox_inches="tight", dpi=150)
                print(f"Saved final graph to {output_img_path}")
            elif agent:
                print("Generating and saving final vessel timeline plot...")
                fig_new, ax_new = plt.subplots(figsize=(12, 6))
                fig_new.subplots_adjust(left=0.08, right=0.80)
                agent.fig = fig_new
                agent.ax_track = ax_new
                agent.ax_kde = ax_new.twiny()
                agent._update_tracker_plot()
                fig_new.savefig(output_img_path, bbox_inches="tight", dpi=150)
                plt.close(fig_new)
                print(f"Saved final graph to {output_img_path}")
        except Exception as e:
            print(f"Could not save final graph: {e}")

        # Generate and save a simulated spectrogram of only the tracked signals
        if all_states:
            try:
                print("Generating simulated spectrogram...")
                max_time = agent.current_time
                
                # Filter states to only include those belonging to stable vessels (duration threshold)
                # to eliminate transient noise and false alarm lines
                vessel_history = {}
                for state in all_states:
                    vid = state.vessel_id if state.vessel_id else "Noise"
                    if vid not in vessel_history:
                        vessel_history[vid] = []
                    vessel_history[vid].append(state)
                
                valid_states = []
                min_track_dur = 300.0 if max_time > 600 else 180.0
                for vid, segs in vessel_history.items():
                    total_dur = sum((s.end_time or max_time) - s.start_time for s in segs)
                    if total_dur >= min_track_dur:
                        valid_states.extend(segs)

                t_start = getattr(agent._env, "start_timestamp", 0.0)
                dt = 0.5
                t_grid = np.arange(t_start, max_time, dt)
                f_grid = np.linspace(min_freq, max_freq, 800)
                spec_grid = np.zeros((len(f_grid), len(t_grid)))
                
                for state in valid_states:
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
                        # Use a fixed narrow spread to render sharp, clear, consistent tonal lines
                        spread_t = 8.0
                        
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
                    extent=[t_start, max_time, min_freq, max_freq],
                    cmap="inferno",
                    vmin=-60,
                    vmax=np.max(spec_grid_db)
                )
                fig_sim.colorbar(im_sim, ax=ax_sim, label="Simulated Amplitude (dB)")
                ax_sim.set_xlabel("Real Time (HH:MM:SS)", fontsize=11, fontweight="bold")
                ax_sim.set_ylabel("Frequency (Hz)", fontsize=11, fontweight="bold")
                ax_sim.set_title(f"Simulated Spectrogram (Tracked Vessel Signals Only) - {args.dataset.upper()}", fontsize=13, fontweight="bold")
                ax_sim.grid(True, linestyle="--", alpha=0.3)

                import datetime
                from matplotlib.ticker import FuncFormatter
                def time_formatter(x, pos):
                    try:
                        dt = datetime.datetime.fromtimestamp(x)
                        return dt.strftime("%H:%M:%S")
                    except Exception:
                        return f"{x:.1f}"
                ax_sim.xaxis.set_major_formatter(FuncFormatter(time_formatter))
                
                sim_spectrogram_filename = f"{_ds_prefix}_{args.rl_agent}_simulated_spectrogram.png"
                sim_spectrogram_path = os.path.join(out_dir, sim_spectrogram_filename)
                fig_sim.savefig(sim_spectrogram_path, bbox_inches="tight", dpi=150)
                plt.close(fig_sim)
                print(f"Saved simulated spectrogram to {sim_spectrogram_path}")
            except Exception as e:
                print(f"Could not generate simulated spectrogram: {e}")

        # Generate and save a joint histogram plot of frequency vs amplitude
        if raw_detections:
            try:
                print("Generating joint histogram of frequency and amplitude...")
                # Filter to only show frequencies between 0 and 2.5 kHz in the main pane and marginal histograms
                filtered_dets = [d for d in raw_detections if 0.0 <= d[0] <= 2500.0]
                if not filtered_dets:
                    print("No detections in 0-2500 Hz range for joint histogram.")
                else:
                    freqs_det = [d[0] for d in filtered_dets]
                    amps_det = [d[1] for d in filtered_dets]
                    
                    DARK_BG   = "#1a1a2e"
                    PANEL_BG  = "#16213e"
                    TEXT_COL  = "#e0e0e0"
                    GRID_COL  = "#3a3a5e"
                    
                    # Plotting a beautiful joint histogram
                    fig_joint = plt.figure(figsize=(10, 10), facecolor=DARK_BG)
                    gs = plt.GridSpec(4, 5, hspace=0.15, wspace=0.15,
                                      width_ratios=[1, 1, 1, 0.6, 0.25],
                                      height_ratios=[0.6, 1, 1, 1])
                    
                    # Main 2D density/hist plot
                    ax_joint = fig_joint.add_subplot(gs[1:4, 0:3], facecolor=PANEL_BG)
                    # Marginal histograms
                    ax_marg_x = fig_joint.add_subplot(gs[0, 0:3], sharex=ax_joint, facecolor=PANEL_BG)
                    ax_marg_y = fig_joint.add_subplot(gs[1:4, 3], sharey=ax_joint, facecolor=PANEL_BG)
                    cbar_ax = fig_joint.add_subplot(gs[1:4, 4])
                    
                    # Define y-axis upper limit using a robust percentile to filter out extreme amplitude outliers
                    if len(amps_det) > 10:
                        q99 = np.percentile(amps_det, 99.0)
                        max_amp = q99 if q99 > 0 else max(amps_det)
                    else:
                        max_amp = max(amps_det) if amps_det else 1.0
                    
                    if max_amp <= 0:
                        max_amp = 1.0
                    
                    # Plot hexbin on main axis restricted to the 0-2500 Hz and 0-max_amp range
                    hb = ax_joint.hexbin(freqs_det, amps_det, gridsize=40, cmap="inferno", mincnt=1, edgecolors='none', extent=[0, 2500, 0, max_amp])
                    
                    # Set axis limits
                    ax_joint.set_xlim(0, 2500)
                    ax_joint.set_ylim(0, max_amp * 1.05)
                    
                    # Set axis labels with units
                    ax_joint.set_xlabel("Frequency (Hz)", fontsize=10, fontweight="bold", color=TEXT_COL)
                    ax_joint.set_ylabel("Amplitude (relative units)", fontsize=10, fontweight="bold", color=TEXT_COL)
                    ax_joint.tick_params(colors=TEXT_COL, labelsize=9)
                    
                    # Configure grid ticks and subdivisions (rulers)
                    import matplotlib.ticker as ticker
                    # Frequency ruler (x-axis): major ticks every 500 Hz, minor every 100 Hz to prevent overlap
                    ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(500))
                    ax_joint.xaxis.set_minor_locator(ticker.MultipleLocator(100))
                    
                    # Amplitude ruler (y-axis): reduce scale density using MaxNLocator (max 5 ticks) to show clearly
                    ax_joint.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                    ax_joint.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
                    
                    ax_joint.grid(True, which='major', linestyle="--", alpha=0.4, color=GRID_COL)
                    ax_joint.grid(True, which='minor', linestyle=":", alpha=0.2, color=GRID_COL)
                    
                    # Plot marginal x (frequency) hist
                    ax_marg_x.hist(freqs_det, bins=80, range=(0, 2500), color="#4A90D9", edgecolor="none", alpha=0.8)
                    ax_marg_x.axis('off')
                    
                    # Plot marginal y (amplitude) hist
                    ax_marg_y.hist(amps_det, bins=80, range=(0, max_amp), orientation='horizontal', color="#E67E22", edgecolor="none", alpha=0.8)
                    ax_marg_y.axis('off')
                    
                    # Add vertical colorbar on the right
                    cb = fig_joint.colorbar(hb, cax=cbar_ax, orientation='vertical')
                    cb.set_label("Detection Density (Count)", color=TEXT_COL, fontsize=9, labelpad=10)
                    cb.ax.tick_params(colors=TEXT_COL, labelsize=8)
                    
                    fig_joint.suptitle(f"Joint Frequency & Amplitude Distribution — {args.dataset.replace('_', ' ').upper()}", 
                                       color=TEXT_COL, fontsize=12, fontweight="bold", y=0.95)
                    
                    joint_hist_path = os.path.join(out_dir, f"joint_histogram_{_ds_prefix}.png")
                    fig_joint.savefig(joint_hist_path, bbox_inches="tight", dpi=180, facecolor=DARK_BG)
                    plt.close(fig_joint)
                    print(f"Saved joint histogram to {joint_hist_path}")
            except Exception as e:
                print(f"Could not generate joint histogram: {e}")

if __name__ == "__main__":
    asyncio.run(main())