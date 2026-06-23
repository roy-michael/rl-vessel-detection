# RL Agent Comparison — Croatia 2407_1

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2407_1` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 51,336 | 8,277 | 75 | 0.9% | 1,662 | 260 | 4 |
| **Linear FA (Tile Coding)** | 60,385 | 8,040 | 142 | 1.7% | 1,024 | 286 | 5 |
| **Dyna-Q** | 45,013 | 7,775 | 95 | 1.2% | 1,406 | 244 | 4 |
| **Actor-Critic** | 39,436 | 6,829 | 164 | 2.3% | 1,301 | 192 | 2 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:11:14 – 10:11:01 | 25325 | 225.3 | ±22.2 | 6.0501 | 36.1902 | 256 |
| **Vessel 73** | 09:14:55 – 10:06:02 | 767 | 805.8 | ±84.1 | 0.0078 | 0.0033 | 9 |
| **Vessel 438** | 09:27:22 – 10:01:13 | 636 | 603.0 | ±23.2 | 0.0181 | 0.0151 | 10 |
| **Vessel 1764** | 10:10:13 – 10:11:01 | 190 | 85.8 | ±19.1 | 108.3301 | 80.9743 | 4 |

### Linear FA (Tile Coding) — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:11:14 – 10:06:01 | 23726 | 222.2 | ±23.6 | 8.1533 | 65.4031 | 223 |
| **Vessel 215** | 09:27:22 – 09:51:22 | 710 | 591.5 | ±24.9 | 0.0834 | 0.0642 | 10 |
| **Vessel 804** | 09:38:13 – 09:58:04 | 442 | 658.3 | ±32.0 | 0.0055 | 0.0022 | 5 |
| **Vessel 150** | 09:19:39 – 09:58:59 | 330 | 972.3 | ±91.5 | 0.0081 | 0.0094 | 4 |
| **Vessel 60** | 09:14:55 – 09:58:09 | 214 | 817.7 | ±33.5 | 0.0043 | 0.0001 | 2 |

### Dyna-Q — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:11:14 – 10:09:01 | 27681 | 271.0 | ±24.3 | 3.8190 | 24.7552 | 265 |
| **Vessel 1253** | 10:06:24 – 10:09:01 | 954 | 84.7 | ±18.7 | 52.1661 | 63.5862 | 14 |
| **Vessel 68** | 09:14:55 – 10:03:31 | 517 | 854.3 | ±30.1 | 0.0045 | 0.0014 | 6 |
| **Vessel 946** | 09:41:07 – 10:04:06 | 471 | 783.4 | ±31.5 | 0.0039 | 0.0003 | 3 |

### Actor-Critic — 2 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:11:15 – 10:04:01 | 20549 | 226.5 | ±22.8 | 9.3426 | 56.9582 | 226 |
| **Vessel 1252** | 10:02:53 – 10:04:01 | 359 | 87.7 | ±22.0 | 46.3091 | 77.9433 | 7 |

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

![RL Agent Comparison — Croatia 2407_1](rl_comparison_croatia_2407_1.png)
