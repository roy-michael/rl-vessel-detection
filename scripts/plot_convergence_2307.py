import os
import json
import asyncio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_rl import train_one_episode
from core.agent.rl_agent import RLAgent
from core.agent.policy import QLearningPolicy, SarsaPolicy, DoubleQLearningPolicy, LinearFAPolicy, DynaQPolicy

async def main():
    print("======================================================================")
    print("        GENERATING DEDICATED CONVERGENCE PLOTS FOR 2307 DATASET")
    print("======================================================================\n")

    dataset = "croatia_2307"
    num_episodes = 150
    initial_epsilon = 0.4
    min_epsilon = 0.05
    gamma = 0.85

    policies = {
        "q_learning": QLearningPolicy(alpha=0.15, gamma=gamma),
        "sarsa": SarsaPolicy(alpha=0.15, gamma=gamma),
        "double_q_learning": DoubleQLearningPolicy(alpha=0.15, gamma=gamma),
        "linear_fa": LinearFAPolicy(alpha=0.01, gamma=gamma),
        "dyna_q": DynaQPolicy(alpha=0.15, gamma=gamma, n_planning=20),
    }

    colors = {
        "q_learning":        "#4A90D9",
        "sarsa":             "#E67E22",
        "double_q_learning": "#27AE60",
        "linear_fa":         "#9B59B6",
        "dyna_q":            "#E74C3C",
    }

    names = {
        "q_learning": "Q-Learning",
        "sarsa": "SARSA",
        "double_q_learning": "Double Q-Learning",
        "linear_fa": "Linear FA (Tile Coding)",
        "dyna_q": "Dyna-Q",
    }

    rewards_history = {k: [] for k in policies.keys()}

    for agent_name, policy in policies.items():
        print(f"\n>>> Profiling Policy: {names[agent_name]}")
        rl_agent = RLAgent(policy=policy, epsilon=initial_epsilon)
        
        for ep in range(num_episodes):
            epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (ep / (num_episodes - 1)))
            print(f"  Episode {ep+1}/{num_episodes}... ", end="", flush=True)
            stats = await train_one_episode(ep, rl_agent, epsilon, dataset)
            reward = stats.get('total_reward', 0.0)
            rewards_history[agent_name].append(reward)
            print(f"Reward: {reward:.1f}")

    output_dir = "output/croatia_2307"
    os.makedirs(output_dir, exist_ok=True)

    # Save raw rewards data
    with open(os.path.join(output_dir, "convergence_history.json"), "w") as f:
        json.dump(rewards_history, f, indent=4)

    plt.style.use('dark_background')

    # 1. Combined Convergence Plot
    fig_comb, ax_comb = plt.subplots(figsize=(12, 7))
    fig_comb.patch.set_facecolor('#1a1a2e')
    ax_comb.set_facecolor('#16213e')

    for agent_name, rewards in rewards_history.items():
        ax_comb.plot(
            range(1, num_episodes + 1),
            rewards,
            marker='o',
            markersize=3,
            linewidth=2.0,
            color=colors[agent_name],
            label=names[agent_name]
        )

    ax_comb.set_title("Comparative Policy Convergence (Croatia 2307 - 50 Episodes)", fontsize=14, fontweight='bold', pad=15)
    ax_comb.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax_comb.set_ylabel("Cumulative Episode Reward", fontsize=11, labelpad=8)
    ax_comb.grid(True, color='#2d2d4e', linestyle='--', alpha=0.6)
    legend = ax_comb.legend(facecolor='#1a1a2e', edgecolor='#2d2d4e', loc='lower right')
    plt.setp(legend.get_texts(), color='#e0e0e0')
    
    comb_path = os.path.join(output_dir, "convergence_combined.png")
    fig_comb.savefig(comb_path, dpi=300, bbox_inches='tight', facecolor=fig_comb.get_facecolor())
    plt.close(fig_comb)

    # 2. Individual Subplots Grid
    fig_indiv, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig_indiv.patch.set_facecolor('#1a1a2e')
    axes_flat = axes.flatten()

    for idx, (agent_name, rewards) in enumerate(rewards_history.items()):
        ax = axes_flat[idx]
        ax.set_facecolor('#16213e')
        ax.plot(
            range(1, num_episodes + 1),
            rewards,
            marker='o',
            markersize=4,
            linewidth=2.2,
            color=colors[agent_name]
        )
        ax.set_title(f"{names[agent_name]} Convergence", fontsize=12, fontweight='bold', color=colors[agent_name])
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Cumulative Reward", fontsize=9)
        ax.grid(True, color='#2d2d4e', linestyle='--', alpha=0.5)
        ax.tick_params(colors='#e0e0e0', labelsize=8)

    # Clear the 6th unused subplot in the 3x2 grid
    axes_flat[5].axis('off')

    plt.suptitle("Individual RL Agent Convergence Curves (Croatia 2307)", fontsize=16, fontweight='bold', y=0.98, color='#e0e0e0')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    indiv_path = os.path.join(output_dir, "convergence_individual.png")
    fig_indiv.savefig(indiv_path, dpi=300, bbox_inches='tight', facecolor=fig_indiv.get_facecolor())
    plt.close(fig_indiv)

    print(f"\nSaved combined convergence plot to {comb_path}")
    print(f"Saved individual convergence plots to {indiv_path}")

if __name__ == "__main__":
    asyncio.run(main())
