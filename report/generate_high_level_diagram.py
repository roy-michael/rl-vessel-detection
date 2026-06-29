import base64
import requests
import os

mmd = """---
title: High-Level RL-DSP Tracking Architecture
---
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'background': '#ffffff',
    'primaryColor': '#eff6ff',
    'primaryTextColor': '#1e3a8a',
    'primaryBorderColor': '#bfdbfe',
    'lineColor': '#64748b',
    'secondaryColor': '#f8fafc',
    'tertiaryColor': '#ffffff',
    'fontFamily': 'system-ui, -apple-system, sans-serif'
  }
}}%%
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,stroke-width:2px,rx:15,ry:15,color:#334155;
    classDef highlight fill:#eff6ff,stroke:#3b82f6,stroke-width:2.5px,rx:15,ry:15,color:#1e40af;
    classDef agent fill:#fff7ed,stroke:#ea580c,stroke-width:2.5px,rx:15,ry:15,color:#7c2d12;
    classDef trackers fill:#fff1f2,stroke:#e11d48,stroke-width:2.5px,rx:15,ry:15,color:#9f1239;

    subgraph Layer1 ["1. Signal Processing Layer"]
        Raw(["🌊 Raw Signal"]):::highlight
        Spec(["📊 Spectrogram"]):::highlight
        Peaks(["🧬 Peak Frequencies"]):::highlight
        Raw -->|STFT| Spec
        Spec -->|NMF| Peaks
    end

    subgraph Layer2 ["2. Decision & Control Layer"]
        Orch(["⚙️ DSP Orchestrator"]):::highlight
        Agent(["🧠 RL Agent"]):::agent
        Orch <-->|State & Action| Agent
    end

    subgraph Layer3 ["3. Concurrent Tracking Layer"]
        Tracks(["🛳️ VesselTrackProcessors"]):::trackers
    end

    Peaks -->|Input Peaks| Orch
    Orch -->|Spawn / Associate| Tracks"""

b64 = base64.urlsafe_b64encode(mmd.encode('utf-8')).decode('utf-8')
url = f'https://mermaid.ink/img/{b64}?type=png&bgColor=white&width=2400&scale=2'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'}
r = requests.get(url, headers=headers)

if r.status_code == 200:
    # Save to both paths to ensure all references are correct
    os.makedirs('report/images', exist_ok=True)
    with open('report/images/high_level_architecture.png', 'wb') as f:
        f.write(r.content)
    with open('report/high_level_architecture.png', 'wb') as f:
        f.write(r.content)
    print("Successfully generated high-level diagram PNG.")
else:
    print(f"Failed: {r.status_code} - {r.text}")
