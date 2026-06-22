import matplotlib.pyplot as plt
import os

def draw_reward_flowchart():
    # Setup premium styling
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    
    # Hide axes
    ax.axis('off')
    
    # Box helper
    def draw_box(x, y, w, h, text, box_type, fontsize=9):
        colors = {
            'action': {'face': '#ffe082', 'edge': '#ffb300', 'text': '#000000'},
            'calc': {'face': '#90caf9', 'edge': '#1e88e5', 'text': '#000000'},
            'reward_pos': {'face': '#a5d6a7', 'edge': '#43a047', 'text': '#000000'},
            'reward_neg': {'face': '#ef9a9a', 'edge': '#e53935', 'text': '#000000'},
        }
        c = colors.get(box_type, {'face': '#ffffff', 'edge': '#000000', 'text': '#000000'})
        
        # Draw rectangle
        rect = plt.Rectangle((x, y), w, h, 
                             facecolor=c['face'], edgecolor=c['edge'], linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x + w/2, y + h/2, text, color=c['text'], fontsize=fontsize,
                ha='center', va='center', fontweight='bold', wrap=True)
                
    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color='#e0e0e0', lw=1.5, mutation_scale=12))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, text, color='#e0e0e0', fontsize=8, ha='center', va='bottom')

    # Draw nodes
    # Root Decision
    draw_box(4.0, 7.0, 2.0, 0.8, "Selected Action", "action")
    
    # Action paths
    # Left: REJECT
    draw_box(0.5, 5.0, 2.2, 0.9, "REJECT\nIs there a close,\nstrong signal?", "calc")
    draw_arrow(4.0, 7.4, 1.6, 5.9)
    ax.text(2.6, 6.8, "REJECT", color='#e0e0e0', fontsize=8, fontweight='bold')
    
    # Middle: ASSOCIATE
    draw_box(3.7, 5.0, 2.6, 0.9, "ASSOCIATE\nDo active target\ntracks exist?", "calc")
    draw_arrow(5.0, 7.0, 5.0, 5.9)
    ax.text(5.1, 6.4, "ASSOCIATE", color='#e0e0e0', fontsize=8, fontweight='bold')
    
    # Right: SPAWN
    draw_box(7.3, 5.0, 2.2, 0.9, "SPAWN\nIs there already a\nclose active track?", "calc")
    draw_arrow(6.0, 7.4, 8.4, 5.9)
    ax.text(7.4, 6.8, "SPAWN", color='#e0e0e0', fontsize=8, fontweight='bold')

    # Outcomes for REJECT
    draw_box(0.5, 3.2, 2.2, 0.7, "+2.0: Correct Reject\n(Noise)", "reward_pos", fontsize=8)
    draw_arrow(1.0, 5.0, 1.0, 3.9, "No")
    
    draw_box(0.5, 1.5, 2.2, 0.7, "-10.0: False Negative\n(Miss)", "reward_neg", fontsize=8)
    draw_arrow(2.2, 5.0, 2.2, 2.2, "Yes")
    
    # Outcomes for ASSOCIATE
    draw_box(3.7, 3.2, 2.6, 0.7, "-20.0: Invalid Assoc\n(No Tracks)", "reward_neg", fontsize=8)
    draw_arrow(4.0, 5.0, 4.0, 3.9, "No")
    
    draw_box(3.7, 1.5, 2.6, 1.0, "Check Peak Distance:\n<= 30Hz: +10.0 (Good)\n<= 65Hz: +5.0 (Speed Chg)\n> 65Hz: -15.0 (Mismatch)", "calc", fontsize=7.5)
    draw_arrow(5.5, 5.0, 5.5, 2.5, "Yes")
    
    # Outcomes for SPAWN
    draw_box(7.3, 3.2, 2.2, 0.7, "-10.0: Duplicate Spawn\nPenalty", "reward_neg", fontsize=8)
    draw_arrow(7.8, 5.0, 7.8, 3.9, "Yes")
    
    draw_box(7.3, 1.5, 2.2, 0.8, "Check Tonality:\n>= 0.65: +10.0 (High)\n< 0.65: +5.0 (Med)", "calc", fontsize=8)
    draw_arrow(9.0, 5.0, 9.0, 2.3, "No")

    plt.tight_layout()
    os.makedirs("report/images", exist_ok=True)
    plt.savefig("report/images/reward_flowchart.png", dpi=180, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print("Saved reward_flowchart.png")

def draw_ecosystem_diagram():
    # Setup premium styling
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Box helper
    def draw_box(x, y, w, h, text, box_type, fontsize=9):
        colors = {
            'env': {'face': '#b2dfdb', 'edge': '#00695c', 'text': '#000000'},
            'dsp': {'face': '#f8bbd0', 'edge': '#ad1457', 'text': '#000000'},
            'rl': {'face': '#e1bee7', 'edge': '#6a1b9a', 'text': '#000000'},
            'data': {'face': '#ffe0b2', 'edge': '#ef6c00', 'text': '#000000'},
        }
        c = colors.get(box_type, {'face': '#ffffff', 'edge': '#000000', 'text': '#000000'})
        rect = plt.Rectangle((x, y), w, h, 
                             facecolor=c['face'], edgecolor=c['edge'], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, color=c['text'], fontsize=fontsize,
                ha='center', va='center', fontweight='bold', wrap=True)
                
    # Arrow helper
    def draw_arrow(x1, y1, x2, y2, text="", style="-|>"):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color='#e0e0e0', lw=1.5, mutation_scale=12))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, color='#e0e0e0', fontsize=8, ha='center', va='bottom')

    # Draw Containers
    # Environment Wrapper Container
    rect_env = plt.Rectangle((0.2, 3.4), 9.6, 4.3, facecolor='none', edgecolor='#00695c', linestyle='--', linewidth=1.5)
    ax.add_patch(rect_env)
    ax.text(0.4, 7.4, "Acoustic Environment Wrapper", color='#b2dfdb', fontsize=11, fontweight='bold')

    # Data
    draw_box(0.5, 5.8, 2.2, 1.0, "Raw Audio\n(STFT Caching)", "data")
    draw_box(0.5, 4.2, 2.2, 1.0, "Ambient Noise\n(NMF Model)", "data")
    
    # Dispatcher
    draw_box(3.5, 5.0, 2.5, 1.2, "DispatcherAgent\n(NMF Peak Extraction & \nSpectral Clustering)", "dsp")
    
    # Gym Wrapper
    draw_box(7.0, 5.0, 2.4, 1.2, "VesselTrackingRLEnv\n(Gym / MDP Interface)", "env")
    
    # Flows inside environment
    draw_arrow(2.7, 6.3, 3.5, 5.9)
    draw_arrow(2.7, 4.7, 3.5, 5.3)
    draw_arrow(6.0, 5.6, 7.0, 5.6, "Extracted Peaks")
    
    # RL Brain
    draw_box(3.5, 2.0, 2.5, 1.0, "RL Tracking Agent\n(Double Q-Learning\n/ Dyna-Q)", "rl")
    
    # Connect environment to RL Brain
    draw_arrow(8.2, 5.0, 8.2, 2.5, style="<-")
    ax.text(8.3, 3.5, "State s_t\nReward r_t", color='#e0e0e0', fontsize=8, ha='left')
    draw_arrow(8.2, 2.5, 6.0, 2.5, style="-")
    
    # Connect RL Brain back with Action
    draw_arrow(3.5, 2.5, 1.6, 2.5, "Action a_t", style="-")
    draw_arrow(1.6, 2.5, 1.6, 5.0, style="-|>")
    
    # Child agents
    draw_box(7.0, 0.5, 2.4, 1.0, "SignalProcessorAgent 1\n(Vessel Track A)", "dsp")
    draw_box(3.5, 0.5, 2.5, 1.0, "SignalProcessorAgent 2\n(Vessel Track B)", "dsp")
    
    # Connect Dispatcher to Child Agents
    draw_arrow(4.75, 5.0, 4.75, 1.5, "SPAWN / ASSOCIATE")
    draw_arrow(4.75, 1.5, 8.2, 1.5, style="-|>")
    draw_arrow(4.75, 1.5, 4.75, 1.5, style="-|>")
    
    plt.tight_layout()
    os.makedirs("report/images", exist_ok=True)
    plt.savefig("report/images/ecosystem_flowchart.png", dpi=180, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print("Saved ecosystem_flowchart.png")

if __name__ == "__main__":
    draw_reward_flowchart()
    draw_ecosystem_diagram()
