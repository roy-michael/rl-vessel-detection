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
    croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
    file_dir = f"{croatia_base_dir}/2507_1"
    
    min_freq = 40
    max_freq = 2000
    n_fft = 16 * 1024
    hop_length = n_fft // 2 # int(1 * 1024)
    window_sec = 15.0
    n_components = 8

    # Parameters for component clustering and variance scaling
    proximity_threshold_hz = 65.0
    association_threshold_hz = 80.0
    peak_spread_window_bins = 15
    variance_multiplier = 1.5
    min_variance_floor = 25.0
    consolidation_threshold_hz = 65.0

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
                
                log(f"\n>>> {vid} [{dominance_status}] (Total Active Duration: {dur:.1f}s):")
                for s_idx, state in enumerate(segs_sorted):
                    status = "ACTIVE" if state in active_states else "COMPLETED"
                    s_dur = (state.end_time or agent.current_time) - state.start_time
                    log(f"  Speed Stage {s_idx+1} [{status}]:")
                    log(f"    Interval:       {state.start_time:.1f}s - {(state.end_time or agent.current_time):.1f}s (Duration: {s_dur:.1f}s)")
                    log(f"    Mean Frequency: {state.mean_frequency:.1f} Hz")
                    log(f"    Total Variance: {state.total_variance:.1f} Hz^2 (Std Dev: {np.sqrt(state.total_variance):.1f} Hz)")
                log("-" * 55)
        log("="*70 + "\n")

        # Save the textual report to a file
        try:
            output_report_path = os.path.join("output", "vessel_detection_report.txt")
            with open(output_report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"Saved textual report to {output_report_path}")
        except Exception as e:
            print(f"Could not save textual report: {e}")

        # Save the matplotlib figure to an image
        if agent and hasattr(agent, 'fig') and plt.fignum_exists(agent.fig.number):
            try:
                output_img_path = os.path.join("output", "vessel_detection_timeline.png")
                agent.fig.savefig(output_img_path, bbox_inches="tight", dpi=150)
                print(f"Saved final graph to {output_img_path}")
            except Exception as e:
                print(f"Could not save figure image: {e}")

if __name__ == "__main__":
    asyncio.run(main())