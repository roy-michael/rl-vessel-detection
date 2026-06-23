import base64
import requests

mmd = """---
title: RL Tracking Feedback Loop
---
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'sans-serif'}}}%%
graph TB
    classDef engine fill:#2C3E50,stroke:#18BC9C,stroke-width:4px,color:#ffffff;
    classDef env fill:#E8F8F5,stroke:#117A65,stroke-width:4px,color:#117A65;
    classDef data fill:#FEF9E7,stroke:#D4AC0D,stroke-width:3px,color:#7D6608;
    classDef action fill:#FDEDEC,stroke:#E74C3C,stroke-width:3px,color:#922B21;

    Agent([🧠 RL Agent<br/>Decision Engine]):::engine
    Env[[🌊 Tracking Environment<br/>MDP Wrapper]]:::env

    subgraph The Feedback Loop
        direction LR
        Action([⚡ Action<br/>SPAWN, ASSOCIATE, REJECT]):::action
        State([📊 State<br/>Distance, Amplitude, Tonality]):::data
        Reward([💰 Reward<br/>+10 Good, -10 Penalty]):::data
    end

    Agent -->|Executes| Action
    Action -->|Applied to| Env
    
    Env -->|Observes| State
    Env -->|Calculates| Reward
    
    State -->|Input to| Agent
    Reward -->|Optimizes| Agent"""

b64 = base64.urlsafe_b64encode(mmd.encode('utf-8')).decode('utf-8')
url = f'https://mermaid.ink/img/{b64}?type=png&bgColor=white&width=2400&scale=2'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    with open('report/high_level_architecture.png', 'wb') as f:
        f.write(r.content)
    print("Successfully generated high-level diagram PNG.")
else:
    print(f"Failed: {r.status_code} - {r.text}")
