# RL Agent Comparison — Scooter

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Scooter` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 44,510 | 5,792 | 31 | 0.5% | 685 | 88 | 8 |
| **Linear FA (Tile Coding)** | 54,145 | 6,421 | 52 | 0.8% | 509 | 82 | 5 |
| **Dyna-Q** | 43,035 | 5,966 | 44 | 0.7% | 626 | 89 | 2 |
| **Actor-Critic** | 37,319 | 5,636 | 116 | 2.0% | 768 | 57 | 2 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 8 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 06:00:14 – 06:31:41 | 13196 | 466.3 | ±22.1 | 0.0129 | 0.0086 | 95 |
| **Vessel 12** | 06:00:20 – 06:32:13 | 2442 | 563.0 | ±43.3 | 0.0065 | 0.0077 | 18 |
| **Vessel 685** | 06:26:16 – 06:32:46 | 389 | 513.3 | ±27.6 | 0.0056 | 0.0018 | 3 |
| **Vessel 621** | 06:15:54 – 06:30:47 | 342 | 599.1 | ±23.3 | 0.0070 | 0.0047 | 3 |
| **Vessel 639** | 06:24:31 – 06:34:15 | 319 | 939.1 | ±29.7 | 0.0194 | 0.0088 | 3 |
| **Vessel 634** | 06:23:03 – 06:37:01 | 315 | 667.4 | ±27.7 | 0.0047 | 0.0022 | 4 |
| **Vessel 650** | 06:25:17 – 06:37:01 | 307 | 764.9 | ±54.0 | 0.0054 | 0.0037 | 4 |
| **Vessel 644** | 06:24:39 – 06:34:19 | 181 | 1025.7 | ±30.5 | 0.0141 | 0.0026 | 3 |

### Linear FA (Tile Coding) — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 06:00:14 – 06:33:54 | 11837 | 466.0 | ±22.2 | 0.0157 | 0.0124 | 90 |
| **Vessel 31** | 06:01:55 – 06:40:01 | 5036 | 550.0 | ±26.8 | 0.0069 | 0.0109 | 25 |
| **Vessel 549** | 06:33:16 – 06:40:01 | 459 | 668.2 | ±30.6 | 0.0080 | 0.0024 | 4 |
| **Vessel 490** | 06:26:32 – 06:29:35 | 189 | 1145.3 | ±11.2 | 0.0308 | 0.0119 | 3 |
| **Vessel 514** | 06:27:35 – 06:40:01 | 189 | 763.5 | ±24.7 | 0.0081 | 0.0037 | 3 |

### Dyna-Q — 2 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 06:00:14 – 06:35:48 | 14913 | 471.4 | ±22.4 | 0.0140 | 0.0115 | 109 |
| **Vessel 253** | 06:04:51 – 06:42:01 | 2799 | 551.8 | ±25.0 | 0.0052 | 0.0046 | 18 |

### Actor-Critic — 2 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 06:00:14 – 06:42:01 | 16696 | 464.2 | ±22.5 | 0.0134 | 0.0158 | 127 |
| **Vessel 34** | 06:01:19 – 06:38:04 | 1763 | 602.7 | ±50.3 | 0.0110 | 0.0098 | 16 |

---

## 3. Agent Descriptions

| Agent | Course Lesson | Key Mechanism |
| :--- | :--- | :--- |
| **Double Q-Learning** | Lesson 5 ext. | Two Q-tables decouple action *selection* from *evaluation*, removing maximisation bias |
| **Linear FA (Tile Coding)** | Lesson 6 | Continuous state → 2,048-dim tile-coded feature vector; weight vector `w` per action |
| **Dyna-Q** | Lesson 7 | Q-Learning + 20 simulated planning steps per real step using a learned transition model |
| **Actor-Critic** | Custom | Policy Gradient + Value Function estimation with Softmax action selection |

---

## 4. Comparison Figure

![RL Agent Comparison — Scooter](rl_comparison_scooter.png)
