"""
compare_rl_agents.py
====================
End-to-end pipeline:
  - Trains all 5 RL agents on a single training dataset
  - Evaluates all 5 agents on one or more verification datasets
  - Produces a separate comparison report + figure per verification dataset

Outputs (per eval dataset):
  output/rl_comparison_{eval_dataset}.png
  output/rl_comparison_{eval_dataset}.md

Usage:
  python compare_rl_agents.py
  python compare_rl_agents.py --train-dataset croatia --eval-datasets croatia_2507_2 croatia_2407_1 croatia_2407_2
  python compare_rl_agents.py --skip-training
"""
import os
import sys
import re
import json
import argparse
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON = sys.executable

AGENTS = [
    ("q_learning",        "Q-Learning"),
    ("sarsa",             "SARSA"),
    ("double_q_learning", "Double Q-Learning"),
    ("linear_fa",         "Linear FA\n(Tile Coding)"),
    ("dyna_q",            "Dyna-Q"),
]

COLORS = {
    "q_learning":        "#4A90D9",
    "sarsa":             "#E67E22",
    "double_q_learning": "#27AE60",
    "linear_fa":         "#9B59B6",
    "dyna_q":            "#E74C3C",
}

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
TEXT_COL  = "#e0e0e0"
GRID_COL  = "#2d2d4e"
HDR_BG    = "#0f3460"
ROW_ALT   = "#1e2a4a"

DATASET_LABELS = {
    "croatia":        "Croatia 2507_1",
    "croatia_2507_2": "Croatia 2507_2",
    "croatia_2407_1": "Croatia 2407_1",
    "croatia_2407_2": "Croatia 2407_2",
    "croatia_2307":   "Croatia 2307",
    "scooter":        "Scooter",
}


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def run_step(cmd: list, cwd: str) -> tuple:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Report filename convention
# ---------------------------------------------------------------------------

DS_PREFIX = {
    "croatia":        "croatia_2507_1",
    "croatia_2507_2": "croatia_2507_2",
    "croatia_2407_1": "croatia_2407_1",
    "croatia_2407_2": "croatia_2407_2",
    "croatia_2307":   "croatia_2307",
    "scooter":        "scooter",
}

def report_path_for(agent_id: str, eval_dataset: str) -> str:
    prefix = DS_PREFIX.get(eval_dataset, eval_dataset)
    if eval_dataset == "scooter":
        return f"output/{prefix}/scooter_{agent_id}_vessel_detection_report.txt"
    return f"output/{prefix}/{prefix}_{agent_id}_report.txt"


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def parse_report(report_path: str) -> dict:
    metrics = {
        "total_reward":       0.0,
        "good_assoc":         0,
        "bad_assoc":          0,
        "correct_spawn":      0,
        "dup_spawn":          0,
        "correct_reject":     0,
        "n_dominant_vessels": 0,
    }
    if not os.path.exists(report_path):
        return metrics

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"Cumulative Reward:\s+([-\d.]+)", content)
    if m:
        metrics["total_reward"] = float(m.group(1))

    m = re.search(r"Outcome Statuses:\s+(\{[^}]+\})", content)
    if m:
        try:
            raw = m.group(1).replace("'", '"')
            statuses = json.loads(raw)
            metrics["good_assoc"]     = statuses.get("good_association", 0)
            metrics["bad_assoc"]      = statuses.get("bad_association_mismatch", 0)
            metrics["correct_spawn"]  = statuses.get("correct_spawn", 0)
            metrics["dup_spawn"]      = statuses.get("duplicate_spawn_penalty", 0)
            metrics["correct_reject"] = statuses.get("correct_reject", 0)
        except Exception:
            pass

    # Each dominant vessel appears once in the summary table header and once in
    # the detailed section, so we halve the count.
    metrics["n_dominant_vessels"] = len(re.findall(r"DOMINANT TARGET", content)) // 2
    return metrics


def parse_vessels(report_path: str) -> list:
    """Parse per-vessel details from a report. Returns list of dicts, dominant targets only."""
    vessels = []
    if not os.path.exists(report_path):
        return vessels

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?=>>> Vessel)", content)
    for block in blocks:
        if "[DOMINANT TARGET]" not in block:
            continue

        m_hdr = re.match(
            r">>> (Vessel \d+) \[DOMINANT TARGET\] "
            r"\(Total Active Duration:\s*([\d.]+)s,\s*Weighted Mean Amp:\s*([\d.]+)\)",
            block.strip()
        )
        if not m_hdr:
            continue

        vid      = m_hdr.group(1)
        duration = float(m_hdr.group(2))
        mean_amp = float(m_hdr.group(3))

        m_tw = re.search(r"Absolute Active Time Window:\s*([\d.]+)s\s*-\s*([\d.]+)s", block)
        start_t = float(m_tw.group(1)) if m_tw else 0.0
        end_t   = float(m_tw.group(2)) if m_tw else 0.0

        stage_freqs = [float(x) for x in re.findall(r"Mean Frequency:\s*([\d.]+)\s*Hz", block)]
        stage_amps  = [float(x) for x in re.findall(r"Mean Amplitude:\s*([\d.]+)", block)]
        stage_stds  = [float(x) for x in re.findall(r"Std Dev:\s*([\d.]+)\s*Hz", block)]

        if not stage_freqs:
            continue

        vessels.append({
            "id":        vid,
            "duration":  duration,
            "start_t":   start_t,
            "end_t":     end_t,
            "mean_amp":  mean_amp,
            "amp_std":   float(np.std(stage_amps)) if len(stage_amps) > 1 else 0.0,
            "mean_freq": float(np.mean(stage_freqs)),
            "freq_std":  float(np.mean(stage_stds)) if stage_stds else 0.0,
            "n_stages":  len(stage_freqs),
        })

    vessels.sort(key=lambda v: v["duration"], reverse=True)
    return vessels


def parse_episode_rewards(stdout: str) -> list:
    return [
        float(m.group(1))
        for m in re.finditer(r"Episode \d+ Completed \| Cumulative Reward:\s*([-\d.]+)", stdout)
    ]


# ---------------------------------------------------------------------------
# Comparison Figure  (6 panels, dark theme)
# ---------------------------------------------------------------------------

def styled_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=7)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.grid(True, linestyle="--", alpha=0.22, color=GRID_COL)


def build_figure(agent_ids, agent_labels, all_metrics, all_vessels,
                 eval_dataset, episode_rewards):
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    clrs = [COLORS[a] for a in agent_ids]
    x    = np.arange(len(agent_ids))
    bw   = 0.55

    # --- P1: Cumulative Reward ---
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = [m["total_reward"] for m in all_metrics]
    bars = ax1.bar(x, rewards, color=clrs, width=bw, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(agent_labels, fontsize=7.5)
    ax1.set_ylabel("Cumulative Reward")
    styled_ax(ax1, "Cumulative Reward")
    for bar, val in zip(bars, rewards):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + abs(bar.get_height()) * 0.02,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=6.5, color=TEXT_COL)

    # --- P2: Good vs Bad Associations ---
    ax2 = fig.add_subplot(gs[0, 1])
    good = [m["good_assoc"] for m in all_metrics]
    bad  = [m["bad_assoc"]  for m in all_metrics]
    ax2.bar(x - bw/4, good, width=bw/2, color="#27AE60", label="Good Assoc", edgecolor="white", linewidth=0.4)
    ax2.bar(x + bw/4, bad,  width=bw/2, color="#E74C3C", label="Bad Assoc",  edgecolor="white", linewidth=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels(agent_labels, fontsize=7.5)
    ax2.set_ylabel("Count"); ax2.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)
    styled_ax(ax2, "Association Quality")

    # --- P3: Vessels Found ---
    ax3 = fig.add_subplot(gs[0, 2])
    n_v = [m["n_dominant_vessels"] for m in all_metrics]
    bars3 = ax3.bar(x, n_v, color=clrs, width=bw, edgecolor="white", linewidth=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(agent_labels, fontsize=7.5)
    ax3.set_ylabel("Count")
    styled_ax(ax3, "Dominant Vessel Tracks Found")
    for bar, val in zip(bars3, n_v):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                 str(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT_COL)

    # --- P4: Vessel Frequency Distribution bubble chart ---
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(PANEL_BG)
    for spine in ax4.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax4.grid(True, linestyle="--", alpha=0.22, color=GRID_COL)

    y_offsets = {aid: i * 0.65 for i, aid in enumerate(agent_ids)}
    legend_patches = []

    for aid, label, vessels in zip(agent_ids, agent_labels, all_vessels):
        yo    = y_offsets[aid]
        color = COLORS[aid]
        for v in vessels:
            size  = min(700, max(60, v["duration"] * 0.4))
            alpha = min(0.95, max(0.35, v["mean_amp"] * 25))
            ax4.scatter(v["mean_freq"], yo, s=size, color=color, alpha=alpha,
                        edgecolors="white", linewidths=0.5, zorder=3)
            ax4.errorbar(v["mean_freq"], yo, xerr=v["freq_std"], fmt="none",
                         ecolor=color, elinewidth=1.2, capsize=3, alpha=0.6, zorder=2)
            ax4.text(v["mean_freq"], yo + 0.22,
                     v["id"].replace("Vessel ", "V"),
                     ha="center", va="bottom", fontsize=6, color=TEXT_COL)
        legend_patches.append(mpatches.Patch(color=color, label=label.replace("\n", " ")))

    ax4.set_yticks(list(y_offsets.values()))
    ax4.set_yticklabels([a.replace("\n", " ") for a in agent_ids], fontsize=8, color=TEXT_COL)
    ax4.set_xlabel("Mean Frequency (Hz)", fontsize=9, color=TEXT_COL)
    ax4.tick_params(colors=TEXT_COL)
    ax4.set_title(
        "Detected Vessel Frequency Distribution  "
        "[bubble = duration, opacity = mean amplitude, bars = freq std dev]",
        color=TEXT_COL, fontsize=10, fontweight="bold", pad=7
    )
    ax4.legend(handles=legend_patches, fontsize=7.5, facecolor=PANEL_BG,
               labelcolor=TEXT_COL, framealpha=0.6, loc="upper right")

    # --- P5: Training Convergence ---
    ax5 = fig.add_subplot(gs[2, 0])
    any_curve = False
    for aid, label in zip(agent_ids, agent_labels):
        ep_rews = episode_rewards.get(aid, [])
        if ep_rews:
            ax5.plot(range(1, len(ep_rews)+1), ep_rews,
                     color=COLORS[aid], marker="o", markersize=4,
                     label=label.replace("\n", " "), linewidth=1.5)
            any_curve = True
    if not any_curve:
        ax5.text(0.5, 0.5, "No training curves\n(--skip-training used)",
                 ha="center", va="center", transform=ax5.transAxes, color=TEXT_COL, fontsize=9)
    ax5.set_xlabel("Training Episode"); ax5.set_ylabel("Episode Reward")
    ax5.legend(fontsize=6, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)
    styled_ax(ax5, "Training Convergence")

    # --- P6: Correct Spawns vs Dup Spawns ---
    ax6 = fig.add_subplot(gs[2, 1])
    dup  = [m["dup_spawn"]     for m in all_metrics]
    corr = [m["correct_spawn"] for m in all_metrics]
    ax6.bar(x - bw/4, corr, width=bw/2, color="#2ecc71", label="Correct", edgecolor="white", linewidth=0.4)
    ax6.bar(x + bw/4, dup,  width=bw/2, color="#f39c12", label="Duplicate", edgecolor="white", linewidth=0.4)
    ax6.set_xticks(x); ax6.set_xticklabels(agent_labels, fontsize=7.5)
    ax6.set_ylabel("Count")
    ax6.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.6)
    styled_ax(ax6, "Spawn Actions (Correct vs Duplicate)")

    # --- P7: Efficiency summary table ---
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(PANEL_BG); ax7.axis("off")
    ax7.set_title("Efficiency Summary", color=TEXT_COL, fontsize=10, fontweight="bold", pad=7)
    col_labels = ["Agent", "Reward/\nGood Assoc", "Bad Assoc\nRate %", "Vessels"]
    rows = []
    for aid, label, m in zip(agent_ids, agent_labels, all_metrics):
        total    = m["good_assoc"] + m["bad_assoc"] + 1
        rpa      = m["total_reward"] / max(1, m["good_assoc"])
        bad_rate = 100 * m["bad_assoc"] / total
        rows.append([label.replace("\n", " "), f"{rpa:+.1f}", f"{bad_rate:.1f}%", str(m["n_dominant_vessels"])])
    tbl = ax7.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        if r == 0:
            cell.set_facecolor(HDR_BG); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(ROW_ALT); cell.set_text_props(color=TEXT_COL)
        else:
            cell.set_facecolor(PANEL_BG); cell.set_text_props(color=TEXT_COL)

    ds_label = DATASET_LABELS.get(eval_dataset, eval_dataset)
    fig.suptitle(
        f"RL Agent Comparison  —  Evaluated on {ds_label}",
        color=TEXT_COL, fontsize=14, fontweight="bold", y=1.005
    )
    return fig


# ---------------------------------------------------------------------------
# Markdown Report
# ---------------------------------------------------------------------------

def write_markdown_report(agent_ids, agent_labels, all_metrics, all_vessels,
                           train_dataset, eval_dataset, fig_filename, out_path):
    ds_label     = DATASET_LABELS.get(eval_dataset, eval_dataset)
    train_label  = DATASET_LABELS.get(train_dataset, train_dataset)

    lines = [
        f"# RL Agent Comparison — {ds_label}",
        "",
        f"| | |",
        f"| :--- | :--- |",
        f"| **Training Dataset** | `{train_label}` |",
        f"| **Evaluation Dataset** | `{ds_label}` |",
        "",
        "---",
        "",
        "## 1. Global Performance Metrics",
        "",
        "| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for aid, label, m in zip(agent_ids, agent_labels, all_metrics):
        total    = m["good_assoc"] + m["bad_assoc"] + 1
        bad_pct  = 100 * m["bad_assoc"] / total
        lines.append(
            f"| **{label.replace(chr(10), ' ')}** "
            f"| {m['total_reward']:,.0f} "
            f"| {m['good_assoc']:,} "
            f"| {m['bad_assoc']:,} "
            f"| {bad_pct:.1f}% "
            f"| {m['dup_spawn']:,} "
            f"| {m['correct_spawn']:,} "
            f"| {m['n_dominant_vessels']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Detected Vessel Details — Per Agent",
        "",
        "> **Vessel ID** | **Active Window (s)** | **Duration (s)** | "
        "**Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**",
        "",
    ]

    for aid, label, vessels in zip(agent_ids, agent_labels, all_vessels):
        lbl = label.replace("\n", " ")
        n   = len(vessels)
        lines.append(f"### {lbl} — {n} dominant vessel{'s' if n != 1 else ''} detected")
        lines.append("")
        if not vessels:
            lines.append("_No dominant vessels found._")
            lines.append("")
            continue
        lines.append("| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for v in vessels:
            lines.append(
                f"| **{v['id']}** "
                f"| {v['start_t']:.0f} – {v['end_t']:.0f} "
                f"| {v['duration']:.0f} "
                f"| {v['mean_freq']:.1f} "
                f"| ±{v['freq_std']:.1f} "
                f"| {v['mean_amp']:.4f} "
                f"| {v['amp_std']:.4f} "
                f"| {v['n_stages']} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. Agent Descriptions",
        "",
        "| Agent | Course Lesson | Key Mechanism |",
        "| :--- | :--- | :--- |",
        "| **Q-Learning** | Lesson 5 | Off-policy tabular TD — updates with `max Q(s',a')` regardless of policy |",
        "| **SARSA** | Lesson 5 | On-policy tabular TD — updates with `Q(s', a')` under the *actual* next action |",
        "| **Double Q-Learning** | Lesson 5 ext. | Two Q-tables decouple action *selection* from *evaluation*, removing maximisation bias |",
        "| **Linear FA (Tile Coding)** | Lesson 6 | Continuous state → 2,048-dim tile-coded feature vector; weight vector `w` per action |",
        "| **Dyna-Q** | Lesson 7 | Q-Learning + 20 simulated planning steps per real step using a learned transition model |",
        "",
        "---",
        "",
        "## 4. Comparison Figure",
        "",
        f"![RL Agent Comparison — {ds_label}]({fig_filename})",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", default="croatia",
                        choices=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"])
    parser.add_argument("--eval-datasets", nargs="+",
                        default=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"],
                        choices=["croatia", "croatia_2507_2", "croatia_2407_1", "croatia_2407_2", "croatia_2307", "scooter"],
                        help="One or more verification datasets")
    parser.add_argument("--skip-training", action="store_true",
                        help="Reuse existing policy files, skip training")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(cwd, "output"), exist_ok=True)

    agent_ids    = [a[0] for a in AGENTS]
    agent_labels = [a[1] for a in AGENTS]
    episode_rewards = {aid: [] for aid in agent_ids}

    # -----------------------------------------------------------------------
    # Step 1: Training
    # -----------------------------------------------------------------------
    if not args.skip_training:
        for aid in agent_ids:
            cmd = [PYTHON, "train_rl.py", "--agent", aid, "--dataset", args.train_dataset]
            print(f"\n{'='*60}\n  TRAINING: {aid} on {args.train_dataset}\n{'='*60}")
            rc, out, err = run_step(cmd, cwd)
            print(out[-3000:] if len(out) > 3000 else out)
            if rc != 0:
                print(f"[ERROR] {err[-500:]}")
            episode_rewards[aid] = parse_episode_rewards(out)
    else:
        print("Skipping training (--skip-training).")

    # -----------------------------------------------------------------------
    # Step 2 + 3: For each eval dataset — evaluate all agents then report
    # -----------------------------------------------------------------------
    for eval_ds in args.eval_datasets:
        print(f"\n{'#'*70}")
        print(f"  VERIFICATION DATASET: {eval_ds}")
        print(f"{'#'*70}")

        # Evaluate all agents on this dataset
        for aid in agent_ids:
            cmd = [
                PYTHON, "vessel_tracker_rl.py",
                "--rl-agent", aid,
                "--policy-dataset", args.train_dataset,
                "--dataset", eval_ds,
                "--headless",
            ]
            print(f"\n  -- Evaluating {aid} --")
            rc, out, err = run_step(cmd, cwd)
            print(out[-1500:] if len(out) > 1500 else out)
            if rc != 0:
                print(f"  [ERROR] {err[-400:]}")

        # Parse results
        all_metrics = []
        all_vessels = []
        print(f"\n  Results for {eval_ds}:")
        for aid in agent_ids:
            rp = os.path.join(cwd, report_path_for(aid, eval_ds))
            m  = parse_report(rp)
            v  = parse_vessels(rp)
            all_metrics.append(m)
            all_vessels.append(v)
            print(
                f"  [{aid:20s}]  reward={m['total_reward']:>10,.0f}  "
                f"good={m['good_assoc']:>6,}  bad={m['bad_assoc']:>6,}  "
                f"vessels={m['n_dominant_vessels']}  ({len(v)} parsed)"
            )

        # Build figure
        fig = build_figure(agent_ids, agent_labels, all_metrics, all_vessels,
                            eval_ds, episode_rewards)
        prefix = DS_PREFIX.get(eval_ds, eval_ds)
        eval_out_dir = os.path.join(cwd, "output", prefix)
        os.makedirs(eval_out_dir, exist_ok=True)

        fig_fname   = f"rl_comparison_{eval_ds}.png"
        fig_path    = os.path.join(eval_out_dir, fig_fname)
        fig.savefig(fig_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Saved figure : {fig_path}")

        # Write markdown report
        md_fname = f"rl_comparison_{eval_ds}.md"
        md_path  = os.path.join(eval_out_dir, md_fname)
        write_markdown_report(agent_ids, agent_labels, all_metrics, all_vessels,
                               args.train_dataset, eval_ds, fig_fname, md_path)

    print("\n[DONE] All comparisons complete.")


if __name__ == "__main__":
    main()
