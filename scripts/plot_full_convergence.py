import os
import re
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_out_dir = os.path.join(cwd, "output", "croatia_2507_1")
    hist_file = os.path.join(train_out_dir, "convergence_history.json")
    log_file = os.path.join(cwd, "output", "actor_critic_training.log")

    # Load existing history
    if os.path.exists(hist_file):
        with open(hist_file, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # Parse actor_critic log
    ac_rewards = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-16") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
        ac_rewards = [
            float(m.group(1))
            for m in re.finditer(r"Episode \d+ Completed \| Cumulative Reward:\s*([-\d.]+)", content)
        ]
        if ac_rewards:
            history["actor_critic"] = ac_rewards

    # Save updated history
    with open(hist_file, "w") as f:
        json.dump(history, f, indent=4)

    # Plot
    COLORS = {
        "q_learning":        "#4A90D9",
        "sarsa":             "#E67E22",
        "double_q_learning": "#27AE60",
        "linear_fa":         "#9B59B6",
        "dyna_q":            "#E74C3C",
        "actor_critic":      "#F1C40F",
    }
    LABELS = {
        "q_learning":        "Q-Learning",
        "sarsa":             "SARSA",
        "double_q_learning": "Double Q-Learning",
        "linear_fa":         "Linear FA\n(Tile Coding)",
        "dyna_q":            "Dyna-Q",
        "actor_critic":      "Actor-Critic",
    }

    plt.style.use('dark_background')
    fig_comb, ax_comb = plt.subplots(figsize=(12, 7))
    fig_comb.patch.set_facecolor('#1a1a2e')
    ax_comb.set_facecolor('#16213e')

    for aid, label in LABELS.items():
        rewards = history.get(aid, [])
        if rewards:
            ax_comb.plot(range(1, len(rewards) + 1), rewards, marker='o', markersize=3, linewidth=2.0, color=COLORS[aid], label=label.replace("\n", " "))

    ax_comb.set_title("Comparative Policy Convergence (CROATIA)", fontsize=14, fontweight='bold', pad=15)
    ax_comb.set_xlabel("Training Episode", fontsize=11, labelpad=8)
    ax_comb.set_ylabel("Cumulative Episode Reward", fontsize=11, labelpad=8)
    ax_comb.grid(True, color='#2d2d4e', linestyle='--', alpha=0.6)
    legend = ax_comb.legend(facecolor='#1a1a2e', edgecolor='#2d2d4e', loc='lower right')
    plt.setp(legend.get_texts(), color='#e0e0e0')

    out_path = os.path.join(train_out_dir, "convergence_combined.png")
    fig_comb.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig_comb.get_facecolor())
    plt.close(fig_comb)
    print(f"Convergence plot saved to {out_path}")

if __name__ == "__main__":
    main()
