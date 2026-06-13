# RL Agent Comparison — Croatia 2507_1

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2507_1` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 89,302 | 10,875 | 64 | 0.6% | 1,512 | 297 | 5 |
| **SARSA** | 92,098 | 10,994 | 138 | 1.2% | 1,341 | 331 | 4 |
| **Double Q-Learning** | 85,323 | 10,876 | 193 | 1.7% | 1,496 | 290 | 6 |
| **Linear FA (Tile Coding)** | -45,645 | 4,252 | 334 | 7.3% | 4,262 | 235 | 3 |
| **Dyna-Q** | 87,331 | 10,925 | 67 | 0.6% | 1,532 | 269 | 6 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 6** | 35 – 5403 | 17534 | 217.6 | ±22.8 | 0.0336 | 0.1866 | 210 |
| **Vessel 20** | 56 – 4948 | 1352 | 811.4 | ±27.0 | 0.0058 | 0.0069 | 14 |
| **Vessel 1** | 15 – 4699 | 1310 | 705.8 | ±26.3 | 0.0060 | 0.0035 | 15 |
| **Vessel 352** | 1927 – 4898 | 524 | 846.2 | ±30.1 | 0.0060 | 0.0022 | 7 |
| **Vessel 1514** | 4632 – 5403 | 255 | 584.1 | ±22.1 | 0.0125 | 0.0040 | 3 |

### SARSA — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 5403 | 17198 | 220.6 | ±22.8 | 0.0266 | 0.0922 | 196 |
| **Vessel 15** | 55 – 4730 | 861 | 834.2 | ±29.0 | 0.0063 | 0.0056 | 9 |
| **Vessel 1115** | 3626 – 5300 | 385 | 889.3 | ±42.0 | 0.0057 | 0.0010 | 4 |
| **Vessel 401** | 2631 – 4908 | 329 | 679.4 | ±26.6 | 0.0119 | 0.0043 | 5 |

### Double Q-Learning — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 6** | 41 – 5403 | 26572 | 244.1 | ±23.2 | 0.0255 | 0.0806 | 285 |
| **Vessel 401** | 1901 – 4948 | 856 | 836.5 | ±26.2 | 0.0111 | 0.0047 | 11 |
| **Vessel 493** | 2444 – 5403 | 767 | 558.2 | ±27.3 | 0.0075 | 0.0030 | 9 |
| **Vessel 518** | 2570 – 4931 | 390 | 728.9 | ±24.1 | 0.0096 | 0.0046 | 6 |
| **Vessel 517** | 2567 – 5063 | 319 | 681.7 | ±27.5 | 0.0108 | 0.0044 | 5 |
| **Vessel 1501** | 4647 – 4762 | 199 | 880.3 | ±25.0 | 0.0126 | 0.0030 | 2 |

### Linear FA (Tile Coding) — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 104** | 223 – 5403 | 11005 | 206.0 | ±26.9 | 0.0305 | 0.0920 | 150 |
| **Vessel 25** | 56 – 5403 | 1044 | 708.6 | ±47.1 | 0.0068 | 0.0036 | 12 |
| **Vessel 2446** | 3263 – 4741 | 490 | 856.9 | ±27.2 | 0.0115 | 0.0077 | 7 |

### Dyna-Q — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 4** | 39 – 5403 | 25050 | 231.4 | ±22.7 | 0.0303 | 0.1193 | 271 |
| **Vessel 367** | 1861 – 4931 | 1000 | 796.2 | ±26.6 | 0.0095 | 0.0066 | 12 |
| **Vessel 366** | 1860 – 4707 | 956 | 707.3 | ±25.4 | 0.0080 | 0.0083 | 11 |
| **Vessel 1172** | 3683 – 5315 | 504 | 549.6 | ±24.9 | 0.0069 | 0.0041 | 4 |
| **Vessel 502** | 2554 – 4898 | 490 | 861.6 | ±28.7 | 0.0065 | 0.0032 | 6 |
| **Vessel 1193** | 3706 – 3835 | 232 | 928.2 | ±30.7 | 0.0053 | 0.0004 | 3 |

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

![RL Agent Comparison — Croatia 2507_1](rl_comparison_croatia.png)
