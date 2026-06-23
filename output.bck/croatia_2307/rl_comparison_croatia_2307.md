# RL Agent Comparison — Croatia 2307

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2307` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 20,871 | 3,863 | 29 | 0.7% | 906 | 43 | 3 |
| **Linear FA (Tile Coding)** | 31,321 | 4,385 | 35 | 0.8% | 649 | 55 | 6 |
| **Dyna-Q** | 16,809 | 3,654 | 31 | 0.8% | 901 | 54 | 3 |
| **Actor-Critic** | 11,245 | 3,383 | 43 | 1.3% | 1,029 | 33 | 3 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 10:40:14 – 11:00:11 | 18694 | 468.9 | ±23.0 | 0.0327 | 0.0363 | 168 |
| **Vessel 80** | 10:41:37 – 10:48:00 | 316 | 692.0 | ±27.3 | 0.0174 | 0.0057 | 3 |
| **Vessel 10** | 10:40:22 – 10:45:54 | 199 | 937.3 | ±42.3 | 0.0107 | 0.0014 | 2 |

### Linear FA (Tile Coding) — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 10:40:14 – 11:00:11 | 18925 | 454.7 | ±22.2 | 0.0363 | 0.0412 | 146 |
| **Vessel 16** | 10:40:26 – 10:54:45 | 687 | 623.1 | ±27.0 | 0.0167 | 0.0061 | 8 |
| **Vessel 7** | 10:40:20 – 10:48:20 | 399 | 695.9 | ±29.6 | 0.0108 | 0.0022 | 3 |
| **Vessel 21** | 10:40:40 – 10:48:38 | 370 | 741.9 | ±25.8 | 0.0196 | 0.0071 | 3 |
| **Vessel 617** | 10:54:16 – 11:00:11 | 253 | 488.5 | ±28.7 | 0.0292 | 0.0270 | 4 |
| **Vessel 11** | 10:40:21 – 10:46:24 | 210 | 852.1 | ±30.6 | 0.0092 | 0.0009 | 2 |

### Dyna-Q — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 10:40:14 – 11:00:10 | 17841 | 457.9 | ±22.5 | 0.0337 | 0.0476 | 168 |
| **Vessel 10** | 10:40:36 – 11:00:11 | 1374 | 594.5 | ±25.9 | 0.0212 | 0.0267 | 11 |
| **Vessel 43** | 10:41:36 – 10:50:33 | 502 | 721.8 | ±29.3 | 0.0381 | 0.0436 | 5 |

### Actor-Critic — 3 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 10:40:15 – 11:00:11 | 16159 | 465.6 | ±22.8 | 0.0372 | 0.0534 | 159 |
| **Vessel 874** | 10:53:39 – 10:56:19 | 881 | 475.7 | ±29.2 | 0.0113 | 0.0110 | 11 |
| **Vessel 34** | 10:41:26 – 10:47:21 | 259 | 697.2 | ±27.3 | 0.0116 | 0.0021 | 3 |

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

![RL Agent Comparison — Croatia 2307](rl_comparison_croatia_2307.png)
