# RL Agent Comparison — Croatia 2407_1

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2407_1` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 60,099 | 8,592 | 91 | 1.0% | 1,905 | 273 | 3 |
| **SARSA** | 60,853 | 8,682 | 206 | 2.3% | 1,821 | 253 | 4 |
| **Double Q-Learning** | 56,630 | 8,478 | 184 | 2.1% | 1,526 | 222 | 5 |
| **Linear FA (Tile Coding)** | -168,710 | 333 | 11,468 | 97.2% | 0 | 0 | 1 |
| **Dyna-Q** | 56,052 | 8,427 | 91 | 1.1% | 1,615 | 223 | 4 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 29749 | 228.0 | ±21.5 | 8.5499 | 44.6937 | 305 |
| **Vessel 23** | 113 – 3315 | 323 | 752.7 | ±25.9 | 0.0057 | 0.0021 | 5 |
| **Vessel 1519** | 3185 – 3300 | 229 | 816.2 | ±28.1 | 0.0043 | 0.0004 | 2 |

### SARSA — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 33540 | 244.2 | ±24.5 | 8.9091 | 44.4139 | 324 |
| **Vessel 24** | 58 – 3105 | 401 | 707.7 | ±40.2 | 0.0088 | 0.0047 | 3 |
| **Vessel 1444** | 2974 – 3274 | 244 | 768.8 | ±33.3 | 0.0035 | 0.0003 | 3 |
| **Vessel 708** | 1288 – 3094 | 234 | 597.7 | ±27.1 | 0.0125 | 0.0148 | 4 |

### Double Q-Learning — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 28964 | 234.8 | ±22.4 | 7.7521 | 42.8350 | 273 |
| **Vessel 1356** | 3471 – 3602 | 750 | 86.3 | ±18.4 | 63.9154 | 66.6981 | 11 |
| **Vessel 44** | 154 – 3316 | 550 | 743.3 | ±27.2 | 0.0060 | 0.0018 | 7 |
| **Vessel 73** | 221 – 1932 | 250 | 906.1 | ±90.0 | 0.0065 | 0.0031 | 3 |
| **Vessel 1164** | 2549 – 3316 | 207 | 639.6 | ±29.9 | 0.0133 | 0.0101 | 3 |

### Linear FA (Tile Coding) — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 3587 | 330.5 | ±422.8 | 8.4052 | 0.0000 | 1 |

### Dyna-Q — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 25995 | 201.9 | ±23.9 | 5.7988 | 33.7302 | 246 |
| **Vessel 286** | 674 – 1952 | 1442 | 604.8 | ±23.2 | 0.0244 | 0.0194 | 20 |
| **Vessel 44** | 115 – 3143 | 253 | 823.1 | ±31.1 | 0.0051 | 0.0016 | 4 |
| **Vessel 1022** | 1763 – 3107 | 197 | 686.4 | ±32.6 | 0.0057 | 0.0026 | 3 |

---

## 3. Agent Descriptions

| Agent | Course Lesson | Key Mechanism |
| :--- | :--- | :--- |
| **Q-Learning** | Lesson 5 | Off-policy tabular TD — updates with `max Q(s',a')` regardless of policy |
| **SARSA** | Lesson 5 | On-policy tabular TD — updates with `Q(s', a')` under the *actual* next action |
| **Double Q-Learning** | Lesson 5 ext. | Two Q-tables decouple action *selection* from *evaluation*, removing maximisation bias |
| **Linear FA (Tile Coding)** | Lesson 6 | Continuous state → 2,048-dim tile-coded feature vector; weight vector `w` per action |
| **Dyna-Q** | Lesson 7 | Q-Learning + 20 simulated planning steps per real step using a learned transition model |

---

## 4. Comparison Figure

![RL Agent Comparison — Croatia 2407_1](rl_comparison_croatia_2407_1.png)
