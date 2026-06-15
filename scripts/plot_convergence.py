import os
import asyncio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_rl import train_one_episode
from core.agent.rl_agent import RLAgent
from core.agent.policy import QLearningPolicy, SarsaPolicy, DoubleQLearningPolicy, LinearFAPolicy, DynaQPolicy

async def main():
    print("======================================================================")
    print("               RUNNING RL AGENT CONVERGENCE PROFILE PIPELINE")
    print("======================================================================\n")

    dataset = "croatia"
    num_episodes = 40
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
        print(f"\n>>> Training Agent: {names[agent_name]}")
        rl_agent = RLAgent(policy=policy, epsilon=initial_epsilon)
        
        for ep in range(num_episodes):
            epsilon = max(min_epsilon, initial_epsilon - (initial_epsilon - min_epsilon) * (ep / (num_episodes - 1)))
            print(f"  Episode {ep+1}/{num_episodes} (Epsilon: {epsilon:.3f})... ", end="", flush=True)
            
            stats = await train_one_episode(ep, rl_agent, epsilon, dataset)
            reward = stats.get('total_reward', 0.0)
            rewards_history[agent_name].append(reward)
            print(f"Cumulative Reward: {reward:.1f}")

    # Plotting code
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    for agent_name, rewards in rewards_history.items():
        ax.plot(
            range(1, num_episodes + 1), 
            rewards, 
            marker='o', 
            linewidth=2.5, 
            color=colors[agent_name], 
            label=names[agent_name]
        )

    ax.set_title("RL Agent Policy Convergence (Croatia 2507_1)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Training Episode", fontsize=12, labelpad=10)
    ax.set_ylabel("Cumulative Episode Reward", fontsize=12, labelpad=10)
    ax.set_xticks(range(1, num_episodes + 1))
    ax.grid(True, color='#2d2d4e', linestyle='--', alpha=0.7)
    
    # Customize legend
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='#2d2d4e', loc='lower right')
    plt.setp(legend.get_texts(), color='#e0e0e0')

    # Save outputs
    output_dir = "output/croatia_2507_1"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "agent_convergence.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"\nSaved convergence plot to {plot_path}")

if __name__ == "__main__":
    asyncio.run(main())
