import os
import asyncio

import librosa
import matplotlib.pyplot as plt
import numpy as np

from agent import DispatcherAgent, SignalProcessorAgent
from environment import Environment


async def main():
    base_dir = "D:/RoyStudies/Recordings"
    # base_dir = "D:/RoyStudies/Recordings/Garda_2_26/"
    # base_dir = "D:/RoyStudies/Recordings/AUVExp_1_26/Itamar_AUV"
    croatia_base_dir = f"{base_dir}/Croatia/Ocean Sonics"
    file_dir = f"{croatia_base_dir}/2507_1"
    # file_dir = f"{base_dir}/2_Deep Water/Electric"
    # file_dir = f"{base_dir}/Part1_StraightLine/IcListen6695"    #IcListen6692"
    # file_dir = f"{base_dir}/Part2_Poligon/IcListen6692"    #IcListen6692"
    min_freq = 40
    max_freq = 2000
    n_fft = 16 * 1024
    hop_length = n_fft // 8 # int(1 * 1024)
    window_sec = 15.0
    n_components = 8

    try:
        env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length)
        agent = DispatcherAgent(env, min_freq, max_freq, n_fft, n_components, 2000, window_sec)
        signal_processor = SignalProcessorAgent(agent)

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
        while True:
            await asyncio.sleep(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        print("\n" + "="*70)
        print("                 FINAL VESSEL DETECTION & SPEED REPORT")
        print("="*70)
        states = list(signal_processor.tracker.states)
        active_states = list(signal_processor.tracker.active_states.values())
        all_states = states + active_states
            
        if not all_states:
            print("No stable vessel states tracked during this run.")
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
                
                print(f"\n>>> {vid} [{dominance_status}] (Total Active Duration: {dur:.1f}s):")
                for s_idx, state in enumerate(segs_sorted):
                    status = "ACTIVE" if state in active_states else "COMPLETED"
                    s_dur = (state.end_time or agent.current_time) - state.start_time
                    print(f"  Speed Stage {s_idx+1} [{status}]:")
                    print(f"    Interval:       {state.start_time:.1f}s - {(state.end_time or agent.current_time):.1f}s (Duration: {s_dur:.1f}s)")
                    print(f"    Mean Frequency: {state.mean_frequency:.1f} Hz")
                    print(f"    Total Variance: {state.total_variance:.1f} Hz^2 (Std Dev: {np.sqrt(state.total_variance):.1f} Hz)")
                print("-" * 55)
        print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())