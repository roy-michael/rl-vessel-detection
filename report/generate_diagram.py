import base64
import requests

mmd = """---
title: Multi-Agent Reinforcement Learning Ecosystem
---
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e1e8f0', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f4f4f4', 'nodeTextColor': '#000000', 'fontFamily': 'sans-serif'}}}%%
graph TD
    classDef envLayer fill:#e0f7fa,stroke:#006064,stroke-width:2px,rx:10px,ry:10px,color:#000000;
    classDef dspLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000000;
    classDef rlLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:5px,ry:5px,color:#000000;
    classDef policy fill:#e8eaf6,stroke:#1a237e,stroke-width:2px,color:#000000;
    classDef data fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000;
    classDef action fill:#ffebee,stroke:#b71c1c,stroke-width:2px,stroke-dasharray: 5 5,color:#000000;

    subgraph Environment [Acoustic Environment Wrapper]
        Audio[("Raw Audio<br/>STFT Cache")]:::data
        NMF[("Ambient Noise<br/>Background Model")]:::data
        Dispatcher{"DispatcherAgent<br/>NMF Peak Extraction"}:::dspLayer
        RLEnv["VesselTrackingRLEnv<br/>Gym Interface"]:::envLayer
        
        Audio -->|WAV Frames| Dispatcher
        NMF <-->|Updates / Denoises| Dispatcher
        Dispatcher -->|Extracted Peaks| RLEnv
    end

    subgraph Brain [RL Decision Engine]
        State(("State s_t:<br/>Distance, Amp, Tonal")):::data
        Reward(("Reward r_t:<br/>+10 Good, -10 Bad")):::data
        Agent(("RL Tracking Agent")):::rlLayer
        Policy>"Tracking Policy<br/>Q-Learning / Actor-Critic"]:::policy
        
        RLEnv --> State
        RLEnv --> Reward
        State & Reward --> Agent
        Agent <-->|Q-Value Lookups| Policy
    end

    subgraph Execution [Multi-Agent Trackers]
        Action[["Action a_t:<br/>SPAWN / ASSOCIATE / REJECT"]]:::action
        Child1["SignalProcessorAgent 1<br/>Target A"]:::dspLayer
        Child2["SignalProcessorAgent 2<br/>Target B"]:::dspLayer
        
        Agent --> Action
        Action -->|Passed to| RLEnv
        RLEnv -.->|Execution Delegation| Dispatcher
        Dispatcher -->|ASSOCIATE| Child1
        Dispatcher -->|SPAWN| Child2
    end"""

b64 = base64.urlsafe_b64encode(mmd.encode('utf-8')).decode('utf-8')
url = f'https://mermaid.ink/img/{b64}?type=png&bgColor=white&width=1200&scale=2'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    with open('report/architecture_diagram.png', 'wb') as f:
        f.write(r.content)
    print("Successfully generated diagram PNG.")
else:
    print(f"Failed: {r.status_code} - {r.text}")
