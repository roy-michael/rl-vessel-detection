import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt

from core.environment import Environment
from core.agent import DispatcherAgent, SignalProcessorAgent
from core.rl_agent import QLearningAgent, SarsaAgent
from core.rl_env import VesselTrackingRLEnv

BASE_DIR = os.environ.get("RECORDINGS_DIR", "D:/RoyStudies/Recordings")

async def run_simulation_and_collect_tracks(dataset, rl_agent_name, policy_dataset, max_files):
    """
    Runs a headless simulation and collects raw peak detections and final tracked vessel states.
    """
    if dataset == "scooter":
        file_dir = f"{BASE_DIR}/DepartmentalCruise-2025-06-12/icListen/wav"
        min_freq = 400
    elif dataset == "croatia_2507_2":
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2507_2"
        min_freq = 40
    else:
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2507_1"
        min_freq = 40
    max_freq = 2000
    n_fft = 16 * 1024
    hop_length = n_fft // 2
    window_sec = 15.0
    n_components = 8

    # Parameters matching training/eval pipelines
    proximity_threshold_hz = 25.0
    association_threshold_hz = 30.0
    peak_spread_window_bins = 15
    variance_multiplier = 1.5
    min_variance_floor = 25.0
    consolidation_threshold_hz = 25.0

    env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length, max_files=max_files)
    dispatcher = DispatcherAgent(
        env, min_freq, max_freq, n_fft, n_components, 2000, window_sec, headless=True,
        proximity_threshold_hz=proximity_threshold_hz,
        association_threshold_hz=association_threshold_hz,
        peak_spread_window_bins=peak_spread_window_bins,
        variance_multiplier=variance_multiplier,
        min_variance_floor=min_variance_floor,
        consolidation_threshold_hz=consolidation_threshold_hz,
        min_duration_sec=45.0,
        min_vessel_score=0.50
    )
    signal_processor = dispatcher

    # Load policy if exists
    policy_path = f"output/rl_{rl_agent_name}_{policy_dataset}.json"
    if os.path.exists(policy_path):
        if rl_agent_name == "q_learning":
            q_agent = QLearningAgent(epsilon=0.0)
        else:
            q_agent = SarsaAgent(epsilon=0.0)
        q_agent.load_policy(policy_path)
        rl_env = VesselTrackingRLEnv(signal_processor.tracker)
        
        signal_processor.tracker.q_agent = q_agent
        signal_processor.tracker.rl_env = rl_env
        signal_processor.tracker.rl_epsilon = 0.0
        signal_processor.tracker.rl_stats = {
            'total_reward': 0.0,
            'action_counts': {0: 0, 1: 0, 2: 0},
            'status_counts': {}
        }
        print(f"Loaded policy {policy_path} for {dataset}...")
    else:
        print(f"Warning: Policy {policy_path} not found. Running with default rules.")

    # Record raw detections history via monkey-patching
    raw_detections = []
    original_update_multi = signal_processor.tracker.update_multi

    def custom_update_multi(current_time, detections):
        for det in detections:
            if det['centroid'] > 0:
                raw_detections.append((current_time, det['centroid'], det['amplitude'], det['score']))
        return original_update_multi(current_time, detections)

    signal_processor.tracker.update_multi = custom_update_multi

    await env.start()
    await dispatcher.start()

    while not dispatcher.completed:
        await asyncio.sleep(0.05)

    # Run final consolidation to merge any remaining states at the end of the run
    signal_processor.tracker.consolidate_all_vessels()

    # Collect final tracked states
    states = list(signal_processor.tracker.states)
    active_states = [
        s for s in signal_processor.tracker.active_states.values()
        if (dispatcher.current_time - s.start_time) >= signal_processor.tracker.min_duration_sec
    ]
    all_states = states + active_states

    return all_states, raw_detections, dispatcher.current_time, min_freq

def plot_tracking_process(ax, states, raw_detections, max_time, title, min_freq):
    """
    Plots the raw detections in the background and overlays the final tracked trajectories.
    """
    # Plot raw NMF peak detections as background noise representation
    if raw_detections:
        # For long timelines, filter and downsample raw detections to avoid heavy clutter
        if max_time > 600:
            filtered_dets = [d for d in raw_detections if len(d) > 3 and d[3] >= 0.8]
            downsampled_dets = filtered_dets[::25]
            if downsampled_dets:
                times = [d[0] for d in downsampled_dets]
                freqs = [d[1] for d in downsampled_dets]
                ax.scatter(times, freqs, c='gray', s=1.0, alpha=0.03, label='Raw Detections (Sparse/Tonal)')
        else:
            times = [d[0] for d in raw_detections]
            freqs = [d[1] for d in raw_detections]
            ax.scatter(times, freqs, c='gray', s=3, alpha=0.15, label='Raw Detections')

    # Group states by vessel_id
    vessel_groups = {}
    for state in states:
        vid = state.vessel_id if state.vessel_id else "Noise"
        if vid not in vessel_groups:
            vessel_groups[vid] = []
        vessel_groups[vid].append(state)

    # Compute cumulative durations and filter
    vessel_durations = []
    # Dynamic duration threshold: 300s for long runs to avoid transient clutter, 180s for short runs
    min_track_dur = 300.0 if max_time > 600 else 180.0
    for vid, segs in vessel_groups.items():
        total_duration = sum((s.end_time or max_time) - s.start_time for s in segs)
        if total_duration >= min_track_dur:
            vessel_durations.append((vid, total_duration, segs))

    # Sort vessels by duration descending
    vessel_durations.sort(key=lambda x: x[1], reverse=True)

    for rank, (vessel_id, total_dur, segs) in enumerate(vessel_durations):
        color = plt.cm.tab10(rank % 10)
        
        # Sort segments chronologically
        segs_sorted = sorted(segs, key=lambda s: s.start_time)
        
        for state in segs_sorted:
            t_points = np.linspace(state.start_time, state.end_time or max_time, len(state.frequencies))
            
            # Plot trajectory line and markers
            ax.plot(t_points, state.frequencies, color=color, linewidth=2.0, alpha=0.85)
            ax.scatter(t_points, state.frequencies, color=color, s=8, alpha=0.85, edgecolors='none')

        # Add text label at the end of the last segment for dominant vessels
        if rank < 5 and segs_sorted:
            last_seg = segs_sorted[-1]
            t_last = last_seg.end_time or max_time
            ax.text(t_last + 1.5, last_seg.frequencies[-1], vessel_id, 
                    color=color, fontsize=7, fontweight='bold', 
                    verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0, max_time + 5)
    ax.set_ylim(min_freq - 10, 2000)

async def main():
    if not os.path.exists("output"):
        os.makedirs("output")

    # Set matplotlib style for high aesthetic quality
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    # 2x2 grid figure: Q-learning vs SARSA on Croatia vs Scooter
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Define simulation cases
    # (row, col, agent, dataset, policy_dataset, max_files, title)
    cases = [
        (0, 0, "q_learning", "croatia_2507_2", "croatia_2507_2", None, "A. Q-Learning: Croatia 2507_2 (Training)"),
        (0, 1, "q_learning", "croatia", "croatia_2507_2", None, "B. Q-Learning Verification: Croatia 2507_1 (Verification)"),
        (1, 0, "sarsa", "croatia_2507_2", "croatia_2507_2", None, "C. SARSA: Croatia 2507_2 (Training)"),
        (1, 1, "sarsa", "croatia", "croatia_2507_2", None, "D. SARSA Verification: Croatia 2507_1 (Verification)"),
    ]

    for row, col, agent, dataset, policy_dataset, max_files, title in cases:
        print(f"Running simulation for {agent} on {dataset}...")
        states, raw_detections, max_time, min_freq = await run_simulation_and_collect_tracks(dataset, agent, policy_dataset, max_files)
        plot_tracking_process(axs[row, col], states, raw_detections, max_time, title, min_freq)

    plt.suptitle("Comparison of RL Vessel Tracking Process: Training (Croatia 2507_2) vs. Real Verification (Croatia 2507_1)", 
                 fontsize=14, fontweight='bold', y=0.96)
                 
    output_path = "output/rl_tracking_comparison_visualization.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"\nSuccessfully generated and saved comparison plot to: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
