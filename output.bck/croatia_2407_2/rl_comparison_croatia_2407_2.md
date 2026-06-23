# RL Agent Comparison — Croatia 2407_2

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2407_2` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Double Q-Learning** | 62,910 | 12,923 | 56 | 0.4% | 3,358 | 91 | 8 |
| **Linear FA (Tile Coding)** | 95,252 | 14,527 | 54 | 0.4% | 2,514 | 124 | 8 |
| **Dyna-Q** | 54,254 | 12,465 | 50 | 0.4% | 3,103 | 106 | 8 |
| **Actor-Critic** | 41,015 | 11,811 | 89 | 0.7% | 3,498 | 75 | 7 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Double Q-Learning — 8 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 16:21:14 – 17:21:01 | 56550 | 463.1 | ±21.6 | 0.0248 | 0.0639 | 496 |
| **Vessel 1232** | 16:48:15 – 17:21:01 | 997 | 790.5 | ±32.4 | 0.0513 | 0.1469 | 6 |
| **Vessel 964** | 16:41:03 – 17:19:14 | 604 | 839.6 | ±23.7 | 0.1662 | 0.2395 | 10 |
| **Vessel 3004** | 17:16:51 – 17:21:01 | 534 | 1269.0 | ±33.6 | 0.0072 | 0.0005 | 8 |
| **Vessel 2560** | 17:12:49 – 17:19:40 | 508 | 1224.6 | ±34.3 | 0.0069 | 0.0005 | 6 |
| **Vessel 2506** | 17:10:56 – 17:18:57 | 270 | 563.0 | ±31.7 | 0.0081 | 0.0043 | 4 |
| **Vessel 2199** | 17:03:35 – 17:06:10 | 250 | 661.6 | ±24.4 | 0.0043 | 0.0005 | 2 |
| **Vessel 2799** | 17:14:48 – 17:19:52 | 216 | 1082.4 | ±38.5 | 0.0071 | 0.0007 | 2 |

### Linear FA (Tile Coding) — 8 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 16:21:14 – 17:21:01 | 59327 | 467.6 | ±21.8 | 0.0306 | 0.0710 | 463 |
| **Vessel 1984** | 17:13:41 – 17:19:58 | 932 | 1246.9 | ±32.5 | 0.0071 | 0.0008 | 14 |
| **Vessel 786** | 16:41:18 – 17:05:48 | 726 | 785.1 | ±25.4 | 0.1719 | 0.1964 | 8 |
| **Vessel 1046** | 16:47:36 – 16:49:46 | 627 | 847.4 | ±15.2 | 0.3145 | 0.1260 | 6 |
| **Vessel 1958** | 17:12:32 – 17:21:01 | 587 | 718.4 | ±31.1 | 0.0128 | 0.0039 | 2 |
| **Vessel 594** | 16:33:52 – 17:19:58 | 301 | 1156.5 | ±31.7 | 0.0070 | 0.0009 | 4 |
| **Vessel 591** | 16:33:50 – 17:18:18 | 283 | 964.8 | ±38.3 | 0.0077 | 0.0006 | 4 |
| **Vessel 2496** | 17:19:19 – 17:21:01 | 217 | 1246.8 | ±33.0 | 0.0071 | 0.0008 | 3 |

### Dyna-Q — 8 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 16:21:14 – 17:21:01 | 50795 | 461.6 | ±21.9 | 0.0267 | 0.0432 | 431 |
| **Vessel 1877** | 16:56:42 – 17:12:16 | 4369 | 486.1 | ±21.9 | 0.0169 | 0.0138 | 41 |
| **Vessel 862** | 16:40:53 – 17:21:01 | 1206 | 754.7 | ±29.4 | 0.0561 | 0.0996 | 9 |
| **Vessel 2434** | 17:13:06 – 17:21:01 | 995 | 1235.7 | ±35.1 | 0.0073 | 0.0007 | 15 |
| **Vessel 2197** | 17:07:22 – 17:12:12 | 495 | 495.2 | ±19.7 | 0.0158 | 0.0080 | 6 |
| **Vessel 961** | 16:44:35 – 17:17:20 | 392 | 837.2 | ±30.3 | 0.1738 | 0.1874 | 5 |
| **Vessel 2055** | 17:03:35 – 17:06:52 | 334 | 647.1 | ±24.7 | 0.0041 | 0.0004 | 3 |
| **Vessel 2324** | 17:10:09 – 17:12:12 | 309 | 514.8 | ±21.8 | 0.0248 | 0.0168 | 3 |

### Actor-Critic — 7 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 16:21:14 – 17:21:01 | 55489 | 461.4 | ±21.7 | 0.0332 | 0.0934 | 548 |
| **Vessel 1071** | 16:40:51 – 17:21:01 | 891 | 800.0 | ±28.3 | 0.0880 | 0.1760 | 13 |
| **Vessel 2249** | 17:00:12 – 17:17:57 | 746 | 755.1 | ±48.6 | 0.0106 | 0.0024 | 7 |
| **Vessel 39** | 16:21:30 – 17:07:48 | 723 | 698.1 | ±39.0 | 0.0428 | 0.0540 | 8 |
| **Vessel 2772** | 17:13:05 – 17:21:01 | 653 | 1256.9 | ±33.5 | 0.0070 | 0.0006 | 8 |
| **Vessel 2740** | 17:12:43 – 17:19:59 | 357 | 1183.6 | ±35.0 | 0.0071 | 0.0003 | 4 |
| **Vessel 1201** | 16:44:35 – 17:19:55 | 181 | 928.7 | ±82.6 | 0.0105 | 0.0024 | 2 |

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

![RL Agent Comparison — Croatia 2407_2](rl_comparison_croatia_2407_2.png)
