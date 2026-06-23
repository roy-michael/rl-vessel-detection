# RL Agent Comparison — Croatia 2507_2

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2507_2` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 13,851 | 8,716 | 36 | 0.4% | 3,658 | 81 | 4 |
| **Linear FA (Tile Coding)** | 43,744 | 10,782 | 62 | 0.6% | 3,275 | 107 | 3 |
| **Dyna-Q** | 4,189 | 8,438 | 55 | 0.6% | 3,799 | 81 | 3 |
| **Actor-Critic** | 3,906 | 8,791 | 94 | 1.1% | 4,093 | 67 | 5 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:41:38 – 10:28:01 | 33351 | 112.6 | ±19.7 | 0.9641 | 2.5807 | 395 |
| **Vessel 15** | 09:42:21 – 09:54:32 | 402 | 780.6 | ±20.6 | 0.0855 | 0.0587 | 6 |
| **Vessel 11** | 09:42:18 – 10:09:51 | 365 | 562.5 | ±27.8 | 0.1040 | 0.2232 | 6 |
| **Vessel 596** | 09:53:38 – 10:09:48 | 207 | 803.2 | ±19.1 | 0.1439 | 0.1133 | 3 |

### Linear FA (Tile Coding) — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:41:38 – 10:31:01 | 46558 | 111.4 | ±18.8 | 1.1496 | 2.4585 | 496 |
| **Vessel 510** | 09:53:09 – 10:11:43 | 1361 | 818.9 | ±27.9 | 0.0817 | 0.0716 | 13 |
| **Vessel 9** | 09:42:18 – 09:57:00 | 873 | 696.7 | ±21.1 | 0.1620 | 0.1139 | 8 |

### Dyna-Q — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:41:39 – 10:28:01 | 36045 | 120.6 | ±20.0 | 1.0411 | 2.4470 | 423 |
| **Vessel 625** | 09:53:09 – 10:09:11 | 325 | 802.2 | ±17.7 | 0.1054 | 0.0665 | 5 |
| **Vessel 9** | 09:42:42 – 09:54:42 | 294 | 744.8 | ±24.9 | 0.1004 | 0.0761 | 4 |

### Actor-Critic — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 09:41:19 – 10:31:01 | 39766 | 107.3 | ±19.0 | 1.3422 | 2.8538 | 459 |
| **Vessel 716** | 09:53:27 – 10:11:30 | 969 | 796.3 | ±15.7 | 0.1136 | 0.0623 | 12 |
| **Vessel 16** | 09:42:52 – 09:59:14 | 647 | 729.5 | ±19.2 | 0.1695 | 0.1396 | 6 |
| **Vessel 1901** | 10:07:56 – 10:11:42 | 508 | 870.4 | ±31.1 | 0.0137 | 0.0034 | 3 |
| **Vessel 12** | 09:42:39 – 10:10:27 | 452 | 623.2 | ±27.4 | 0.1869 | 0.1182 | 4 |

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

![RL Agent Comparison — Croatia 2507_2](rl_comparison_croatia_2507_2.png)
