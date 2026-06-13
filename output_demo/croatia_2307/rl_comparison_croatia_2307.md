# RL Agent Comparison — Croatia 2307

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2307` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 93,672 | 16,653 | 28 | 0.2% | 5,701 | 89 | 2 |
| **SARSA** | 90,600 | 16,446 | 50 | 0.3% | 5,812 | 79 | 1 |
| **Double Q-Learning** | 62,272 | 15,023 | 28 | 0.2% | 4,503 | 79 | 1 |
| **Linear FA (Tile Coding)** | -195,280 | 6,538 | 17,376 | 72.7% | 0 | 0 | 1 |
| **Dyna-Q** | 64,277 | 15,121 | 27 | 0.2% | 4,457 | 65 | 1 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 2 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 4262 | 76730 | 118.5 | ±22.5 | 0.1491 | 1.1227 | 849 |
| **Vessel 3847** | 3030 – 4262 | 2169 | 90.4 | ±21.6 | 0.0785 | 0.3303 | 20 |

### SARSA — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 4262 | 77993 | 115.9 | ±22.2 | 0.1645 | 0.8704 | 866 |

### Double Q-Learning — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 4262 | 72587 | 117.6 | ±22.3 | 0.1491 | 0.8229 | 756 |

### Linear FA (Tile Coding) — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 4262 | 4248 | 132.0 | ±111.2 | 0.1774 | 0.0000 | 1 |

### Dyna-Q — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 4262 | 70617 | 119.6 | ±22.3 | 0.1408 | 0.8927 | 730 |

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

![RL Agent Comparison — Croatia 2307](rl_comparison_croatia_2307.png)
