# RL Agent Comparison — Croatia 2507_1

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2507_1` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 78,851 | 10,166 | 36 | 0.4% | 1,404 | 338 | 6 |
| **Linear FA (Tile Coding)** | 87,502 | 10,318 | 168 | 1.6% | 896 | 390 | 6 |
| **Dyna-Q** | 76,805 | 10,384 | 46 | 0.4% | 1,210 | 359 | 6 |
| **Actor-Critic** | 63,026 | 8,837 | 153 | 1.7% | 1,204 | 250 | 5 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 08:01:14 – 09:29:02 | 25968 | 246.9 | ±23.5 | 0.0939 | 0.5388 | 262 |
| **Vessel 456** | 08:41:58 – 09:21:23 | 1009 | 562.9 | ±27.4 | 0.0200 | 0.0293 | 14 |
| **Vessel 448** | 08:41:20 – 09:27:44 | 715 | 716.2 | ±27.7 | 0.0163 | 0.0195 | 11 |
| **Vessel 73** | 08:07:10 – 09:14:18 | 419 | 848.8 | ±30.1 | 0.0078 | 0.0093 | 6 |
| **Vessel 1160** | 09:01:51 – 09:29:02 | 353 | 680.6 | ±27.0 | 0.0115 | 0.0065 | 5 |
| **Vessel 913** | 08:52:29 – 09:19:42 | 332 | 897.5 | ±31.8 | 0.0155 | 0.0142 | 5 |

### Linear FA (Tile Coding) — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 08:01:21 – 09:24:02 | 22591 | 220.5 | ±23.9 | 0.0227 | 0.0643 | 213 |
| **Vessel 57** | 08:07:09 – 09:22:35 | 1582 | 751.6 | ±31.1 | 0.0068 | 0.0050 | 18 |
| **Vessel 264** | 08:35:19 – 09:22:36 | 1231 | 584.7 | ±29.7 | 0.0112 | 0.0097 | 13 |
| **Vessel 787** | 08:53:33 – 09:01:51 | 478 | 870.5 | ±34.4 | 0.0051 | 0.0005 | 4 |
| **Vessel 1015** | 09:10:52 – 09:24:02 | 322 | 370.9 | ±24.5 | 0.0094 | 0.0039 | 4 |
| **Vessel 672** | 08:49:15 – 09:13:18 | 293 | 937.3 | ±50.0 | 0.0139 | 0.0096 | 3 |

### Dyna-Q — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 08:01:14 – 09:31:02 | 26345 | 246.2 | ±24.8 | 0.0384 | 0.4992 | 243 |
| **Vessel 68** | 08:07:10 – 09:22:23 | 2044 | 800.8 | ±28.0 | 0.0063 | 0.0041 | 16 |
| **Vessel 66** | 08:07:07 – 09:19:45 | 1093 | 708.7 | ±27.8 | 0.0057 | 0.0015 | 8 |
| **Vessel 997** | 08:58:02 – 09:18:48 | 374 | 607.4 | ±25.2 | 0.0065 | 0.0010 | 4 |
| **Vessel 1313** | 09:18:34 – 09:19:50 | 255 | 871.1 | ±24.3 | 0.0157 | 0.0034 | 4 |
| **Vessel 1294** | 09:15:39 – 09:22:59 | 222 | 818.8 | ±27.7 | 0.0119 | 0.0053 | 3 |

### Actor-Critic — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 2** | 08:01:16 – 09:25:02 | 19868 | 212.2 | ±26.7 | 0.1051 | 1.6980 | 218 |
| **Vessel 59** | 08:08:02 – 09:14:18 | 1019 | 654.8 | ±34.2 | 0.0138 | 0.0095 | 13 |
| **Vessel 668** | 08:48:03 – 09:25:02 | 594 | 316.5 | ±23.6 | 0.0335 | 0.0635 | 9 |
| **Vessel 110** | 08:13:34 – 09:14:19 | 496 | 740.6 | ±31.1 | 0.0089 | 0.0048 | 7 |
| **Vessel 53** | 08:07:10 – 09:14:21 | 434 | 853.5 | ±35.9 | 0.0058 | 0.0021 | 6 |

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

![RL Agent Comparison — Croatia 2507_1](rl_comparison_croatia.png)
