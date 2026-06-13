# RL Agent Comparison — Croatia 2407_2

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2407_2` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 75,242 | 14,929 | 27 | 0.2% | 6,297 | 80 | 1 |
| **SARSA** | 75,850 | 14,961 | 45 | 0.3% | 6,214 | 77 | 1 |
| **Double Q-Learning** | 40,688 | 13,159 | 53 | 0.4% | 4,526 | 81 | 1 |
| **Linear FA (Tile Coding)** | -158,395 | 7,096 | 15,289 | 68.3% | 0 | 0 | 1 |
| **Dyna-Q** | 39,976 | 13,143 | 36 | 0.3% | 4,630 | 74 | 1 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 16 – 3602 | 69318 | 109.8 | ±23.2 | 0.1840 | 0.6215 | 731 |

### SARSA — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 69286 | 106.4 | ±22.9 | 0.2605 | 1.1618 | 766 |

### Double Q-Learning — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 63335 | 109.5 | ±22.8 | 0.1673 | 0.6271 | 646 |

### Linear FA (Tile Coding) — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 3587 | 120.4 | ±108.4 | 0.2997 | 0.0000 | 1 |

### Dyna-Q — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3602 | 63502 | 106.5 | ±22.5 | 0.2105 | 0.9990 | 666 |

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

![RL Agent Comparison — Croatia 2407_2](rl_comparison_croatia_2407_2.png)
