import base64
import requests

mmd = """---
title: Multi-Agent Tracking Hierarchy
---
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e1e8f0', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f4f4f4', 'nodeTextColor': '#000000', 'fontFamily': 'sans-serif'}}}%%
graph TD
    classDef envLayer fill:#e0f7fa,stroke:#006064,stroke-width:2px,rx:10px,ry:10px,color:#000000;
    classDef dspLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px,color:#000000;
    classDef rlLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,rx:5px,ry:5px,color:#000000;
    classDef policy fill:#e8eaf6,stroke:#1a237e,stroke-width:2px,color:#000000;

    subgraph Environment [Acoustic Environment]
        Dispatcher{"Dispatcher Agent<br/>(Audio Processing & Routing)"}:::dspLayer
        RLEnv["Standardized RL Environment Wrapper"]:::envLayer
        
        Dispatcher -->|Extracted Acoustic Peaks| RLEnv
    end

    subgraph Brain [RL Decision Engine]
        Agent(("Master RL Tracking Agent<br/>(Parent)")):::rlLayer
        Policy>"Tracking Policy"]:::policy
        
        RLEnv -->|Provides States & Rewards| Agent
        Agent <-->|Optimizes| Policy
    end

    subgraph Execution [Execution Layer]
        Processors["Target Trackers<br/>(Child Agents)"]:::dspLayer
        
        Agent -->|Issues Actions: Spawn, Associate, Reject| Processors
        Processors -.->|Track Updates| Dispatcher
    end"""

b64 = base64.urlsafe_b64encode(mmd.encode('utf-8')).decode('utf-8')
url = f'https://mermaid.ink/img/{b64}?type=png&bgColor=white&width=1200&scale=2'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    with open('report/high_level_architecture.png', 'wb') as f:
        f.write(r.content)
    print("Successfully generated high-level diagram PNG.")
else:
    print(f"Failed: {r.status_code} - {r.text}")
