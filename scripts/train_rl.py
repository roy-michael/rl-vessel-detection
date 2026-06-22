import os
import asyncio
import numpy as np

from core.environment import Environment, VesselTrackingRLEnv
from core.agent import DispatcherAgent, SignalProcessorAgent
from core.agent.rl_agent import RLAgent
from core.agent.policy import DoubleQLearningPolicy, LinearFAPolicy, ActorCriticPolicy

# Default dataset directory
BASE_DIR = os.environ.get("RECORDINGS_DIR", "C:/Users/Roy/Recordings")
croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
FILE_DIR = f"{croatia_base_dir}/2507_1"
POLICY_FILE = "output/rl_q_table.json"

async def train_one_episode(episode_idx, rl_agent, epsilon, env):
    """Runs a single training episode (one pass over the WAV files in the directory)."""
    max_freq = 4000
    n_fft = 16 * 1024
    hop_length = n_fft // 2
    window_sec = 15.0
    n_components = 8

    # Parameters for component clustering and variance scaling
    proximity_threshold_hz = 25.0
    association_threshold_hz = 30.0
    peak_spread_window_bins = 15
    variance_multiplier = 1.5
    min_variance_floor = 25.0
    consolidation_threshold_hz = 25.0

    env.reset()

    # We initialize DispatcherAgent in headless mode with threshold parameters
    dispatcher = DispatcherAgent(
        env, env.min_freq, max_freq, n_fft, n_components, 200, window_sec, headless=True,
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

    # Initialize RL environment wrapper
    rl_env = VesselTrackingRLEnv(signal_processor.tracker)

    await env.start()
    
    # Wait for buffer to have something
    first_frame, _ = await env.peek()
    if first_frame is None:
        print("  [Train Error] No audio data available in environment.")
        return 0.0, {}

    # Start agents
    await dispatcher.start()

    # Track learning stats
    total_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}
    status_counts = {}

    # We run the frame-by-frame loop manually or hook it into the tracker updates.
    # To run training, we intercept the update_multi call or run our own loop.
    # To keep code changes minimal, we can modify the tracker inside SignalProcessorAgent
    # to support an RL agent policy. If a Q-learning agent is provided, it will use that,
    # otherwise it will fall back to its heuristic rules.
    # Let's assign our rl_agent and rl_env to the signal_processor's tracker!
    signal_processor.tracker.rl_agent = rl_agent
    signal_processor.tracker.rl_env = rl_env
    signal_processor.tracker.rl_epsilon = epsilon
    signal_processor.tracker.rl_stats = {
        'total_reward': 0.0,
        'action_counts': action_counts,
        'status_counts': status_counts
    }

    # Run until dispatcher is completed
    try:
        while not dispatcher.completed:
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass

    # Clean up tasks if any
    return signal_processor.tracker.rl_stats

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train RL Tracking Agent")
    parser.add_argument("--agent", type=str,
                        choices=["double_q_learning", "linear_fa", "actor_critic"],
                        default="double_q_learning", help="Agent type to train")
    parser.add_argument("--dataset", type=str,
                        choices=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"],
                        default="croatia", help="Dataset to train on")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of audio files to process per episode")
    args = parser.parse_args()

    _ds_prefix = {
        "croatia":         "croatia_2507_1",
        "croatia_2507_2":  "croatia_2507_2",
        "croatia_2407_1":  "croatia_2407_1",
        "croatia_2407_2":  "croatia_2407_2",
        "croatia_2307":    "croatia_2307",
        "scooter":         "scooter",
    }.get(args.dataset, args.dataset)
    policy_dir = os.path.join("output", _ds_prefix)
    os.makedirs(policy_dir, exist_ok=True)

    _linear_fa = (args.agent == "linear_fa")
    policy_file = (
        f"{policy_dir}/rl_{args.agent}_{args.dataset}.npy" if _linear_fa
        else f"{policy_dir}/rl_{args.agent}_{args.dataset}.json"
    )
    agent_titles = {
        "double_q_learning": "DOUBLE Q-LEARNING",
        "linear_fa": "LINEAR FUNCTION APPROXIMATION (Tile Coding)",
        "actor_critic": "ACTOR-CRITIC (Policy Gradient + Value Function)",
    }
    agent_title = agent_titles.get(args.agent, args.agent.upper())

    if args.dataset == "scooter":
        file_dir = f"{BASE_DIR}/DepartmentalCruise-2025-06-12/icListen/wav"
        min_freq = 400
    elif args.dataset == "croatia_2507_2":
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2507_2_joint"
        min_freq = 40
    elif args.dataset == "croatia_2407_1":
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2407_1_600m"
        min_freq = 40
    elif args.dataset == "croatia_2407_2":
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2407_2_snake"
        min_freq = 400
    elif args.dataset == "croatia_2307":
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2307_free"
        min_freq = 400
    else:
        file_dir = f"{BASE_DIR}/Croatia/Ocean Sonics/2507_1_1k"
        min_freq = 40

    max_freq = 4000
    n_fft = 16 * 1024
    hop_length = n_fft // 2
    
    env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length, max_files=args.max_files)

    num_episodes = args.episodes
    initial_epsilon = 1.0
    min_epsilon = 0.05
    alpha = 0.15
    gamma = 0.85

    if args.agent == "double_q_learning":
        policy = DoubleQLearningPolicy(alpha=alpha, gamma=gamma)
    elif args.agent == "linear_fa":
        policy = LinearFAPolicy(alpha=0.01, gamma=gamma)
    elif args.agent == "actor_critic":
        policy = ActorCriticPolicy(alpha_actor=0.05, alpha_critic=0.1, gamma=gamma)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    rl_agent = RLAgent(policy=policy, epsilon=initial_epsilon)

    # If an existing policy file exists, we can bootstrap/load it
    if os.path.exists(policy_file):
        try:
            rl_agent.load_policy(policy_file)
            print("Successfully bootstrapped from existing policy file.\n")
        except Exception as e:
            print(f"Could not load existing policy: {e}. Starting fresh.\n")

    for ep in range(num_episodes):
        # Linear epsilon decay
        if num_episodes > 1:
            epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (ep / (num_episodes - 1)))
        else:
            epsilon = min_epsilon
        print(f"--- Episode {ep+1}/{num_episodes} (Epsilon: {epsilon:.3f}) ---")
        
        stats = await train_one_episode(ep, rl_agent, epsilon, env)
        
        # Display episode results
        action_str = ", ".join(f"A{a}: {count}" for a, count in stats.get('action_counts', {}).items())
        print(f"  Episode {ep+1} Completed | Cumulative Reward: {stats.get('total_reward', 0.0):.1f}")
        print(f"  Actions Taken: {action_str}")
        print(f"  Outcome Statuses: {dict(stats.get('status_counts', {}))}")
        print("-" * 50)

    # Save final policy
    rl_agent.save_policy(policy_file)

    # Print learned state-value summaries
    print("\n======================================================================")
    print("                  TRAINED STATE-VALUE FUNCTION DIAGNOSTICS")
    print("======================================================================")
    print("State Representation: (Distance Bin, Amplitude Bin, Tonality Bin)")
    print("Bins: Dist: 0=Close, 1=Med, 2=Far, 3=Out | Amp/Tonal: 0=Low, 1=Med, 2=High\n")
    print(f"{'State (D,A,T)':<15} | {'V(s) Value':<12} | {'Optimal Action':<15} | {'Q-values [A0, A1, A2]':<25}")
    print("-" * 75)
    
    # Sort states for readable diagnostics
    for state in sorted(rl_agent.q_table.keys()):
        q_vals = rl_agent.q_table[state]
        v_s = rl_agent.get_value(state)
        opt_act = rl_agent.get_best_action(state)
        opt_act_str = "REJECT (Noise)" if opt_act == 0 else "ASSOCIATE" if opt_act == 1 else "SPAWN (New)"
        q_vals_str = "[" + ", ".join(f"{q:.2f}" for q in q_vals) + "]"
        print(f"{str(state):<15} | {v_s:<12.2f} | {opt_act_str:<15} | {q_vals_str:<25}")
    print("======================================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
