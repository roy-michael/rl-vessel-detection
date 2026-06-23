import base64
import requests

mmd = """---
title: Detailed RL Interaction Architecture
---
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'sans-serif'}}}%%
graph TD
    classDef engine fill:#2C3E50,stroke:#18BC9C,stroke-width:4px,color:#ffffff;
    classDef env fill:#E8F8F5,stroke:#117A65,stroke-width:4px,color:#117A65;
    classDef policy fill:#8E44AD,stroke:#5B2C6F,stroke-width:4px,color:#ffffff;
    classDef data fill:#FEF9E7,stroke:#D4AC0D,stroke-width:3px,color:#7D6608;
    classDef action fill:#FDEDEC,stroke:#E74C3C,stroke-width:3px,color:#922B21;

    subgraph The Core RL Engine
        Agent([🧠 RL Tracking Agent]):::engine
        Policy[[📜 Active Policy<br/>Double Q / Actor-Critic]]:::policy
        
        Agent <-->|Optimize / Query| Policy
    end

    subgraph The Environment Boundary
        Environment[[🌊 TrackingMDPEnv<br/>Environment Wrapper]]:::env
    end

    subgraph The Interface Protocol
        direction LR
        State([📊 State s_t<br/>Distance, Amp, Tonal]):::data
        Reward([💰 Reward r_t<br/>Tracking Score]):::data
        Action([⚡ Action a_t<br/>SPAWN, ASSOCIATE, REJECT]):::action
    end

    Environment -->|Outputs| State
    Environment -->|Outputs| Reward
    
    State & Reward -->|Inputs| Agent
    
    Agent -->|Selects| Action
    Action -->|Executes| Environment"""

b64 = base64.urlsafe_b64encode(mmd.encode('utf-8')).decode('utf-8')
url = f'https://mermaid.ink/img/{b64}?type=png&bgColor=white&width=2400&scale=2'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    with open('report/architecture_diagram.png', 'wb') as f:
        f.write(r.content)
    print("Successfully generated diagram PNG.")
else:
    print(f"Failed: {r.status_code} - {r.text}")
