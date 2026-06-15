import os
import json
import asyncio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_rl import train_one_episode
from core.agent.rl_agent import RLAgent
from core.agent.policy import QLearningPolicy

async def main():
    print("======================================================================")
    print("         Q-LEARNING EVALUATION CONVERGENCE PROFILE (CROATIA 2307)")
    print("======================================================================\n")

    dataset = "croatia_2307"
    num_episodes = 150
    initial_epsilon = 0.5
    min_epsilon = 0.01
    gamma = 0.85

    # Initialize policy
    policy = QLearningPolicy(alpha=0.15, gamma=gamma)
    rl_agent = RLAgent(policy=policy, epsilon=initial_epsilon)

    train_rewards = []
    eval_rewards = []

    for ep in range(num_episodes):
        # Epsilon and Alpha decay
        epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (ep / (num_episodes - 1)))
        alpha = max(0.01, 0.20 - (0.20 - 0.01) * (ep / (num_episodes - 1)))
        rl_agent.policy.alpha = alpha
        
        # 1. Run training episode
        print(f"Episode {ep+1}/{num_episodes} | Training (Epsilon: {epsilon:.3f}, Alpha: {alpha:.3f})... ", end="", flush=True)
        train_stats = await train_one_episode(ep, rl_agent, epsilon, dataset)
        train_rew = train_stats.get('total_reward', 0.0)
        train_rewards.append(train_rew)
        print(f"Train Reward: {train_rew:.1f} | ", end="", flush=True)

        # 2. Run evaluation episode (Greedy policy, epsilon = 0)
        # We disable learning by setting epsilon to 0.0 in train_one_episode
        eval_stats = await train_one_episode(ep, rl_agent, 0.0, dataset)
        eval_rew = eval_stats.get('total_reward', 0.0)
        eval_rewards.append(eval_rew)
        print(f"Eval Reward: {eval_rew:.1f}")

    output_dir = "output/croatia_2307"
    os.makedirs(output_dir, exist_ok=True)

    # Save rewards history
    history = {
        "train_rewards": train_rewards,
        "eval_rewards": eval_rewards
    }
    with open(os.path.join(output_dir, "q_learning_convergence_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Plotting code
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    # Plot noisy training rewards
    ax.plot(
        range(1, num_episodes + 1),
        train_rewards,
        color="#E74C3C",
        alpha=0.35,
        linestyle="--",
        label="Training Reward (with Exploration Noise)"
    )

    # Plot clean evaluation rewards
    ax.plot(
        range(1, num_episodes + 1),
        eval_rewards,
        color="#27AE60",
        linewidth=3.0,
        marker='o',
        markersize=4,
        label="Evaluation Reward (Greedy Policy)"
    )

    ax.set_title("Q-Learning Policy Convergence Curve (Croatia 2307)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=11, labelpad=8)
    ax.grid(True, color='#2d2d4e', linestyle='--', alpha=0.6)
    
    # Add trend line for evaluation
    try:
        import numpy as np
        z = np.polyfit(range(1, num_episodes + 1), eval_rewards, 2)
        p = np.poly1d(z)
        ax.plot(range(1, num_episodes + 1), p(range(1, num_episodes + 1)), color="#f1c40f", linestyle=":", linewidth=1.5, label="Policy Trend Line")
    except Exception:
        pass

    legend = ax.legend(facecolor='#1a1a2e', edgecolor='#2d2d4e', loc='lower right')
    plt.setp(legend.get_texts(), color='#e0e0e0')

    plot_path = os.path.join(output_dir, "q_learning_convergence.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nSaved Q-learning convergence plot to {plot_path}")

if __name__ == "__main__":
    asyncio.run(main())
