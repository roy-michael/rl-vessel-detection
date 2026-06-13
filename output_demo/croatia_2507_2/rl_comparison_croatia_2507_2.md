# RL Agent Comparison — Croatia 2507_2

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Croatia 2507_2` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 47,463 | 10,738 | 38 | 0.4% | 4,875 | 82 | 4 |
| **SARSA** | 47,493 | 10,687 | 40 | 0.4% | 4,811 | 81 | 6 |
| **Double Q-Learning** | 20,359 | 9,352 | 82 | 0.9% | 3,688 | 58 | 6 |
| **Linear FA (Tile Coding)** | -217,020 | 1,439 | 15,426 | 91.5% | 0 | 0 | 1 |
| **Dyna-Q** | 23,097 | 9,478 | 37 | 0.4% | 3,621 | 79 | 5 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3002 | 43544 | 111.3 | ±19.1 | 1.1157 | 2.3324 | 498 |
| **Vessel 15** | 96 – 1841 | 1171 | 641.4 | ±22.8 | 0.1246 | 0.1907 | 13 |
| **Vessel 769** | 747 – 1843 | 732 | 798.6 | ±15.0 | 0.1580 | 0.0948 | 9 |
| **Vessel 4074** | 2916 – 3002 | 427 | 84.0 | ±16.1 | 0.4074 | 0.1934 | 6 |

### SARSA — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3002 | 43550 | 113.7 | ±18.7 | 1.0543 | 2.4515 | 510 |
| **Vessel 4057** | 2731 – 3002 | 817 | 83.4 | ±19.1 | 2.3962 | 2.9527 | 11 |
| **Vessel 849** | 831 – 1856 | 730 | 804.7 | ±14.5 | 0.1216 | 0.0639 | 8 |
| **Vessel 656** | 690 – 960 | 618 | 761.9 | ±16.1 | 0.2079 | 0.0944 | 6 |
| **Vessel 575** | 822 – 1754 | 223 | 571.4 | ±19.3 | 0.0932 | 0.0559 | 3 |
| **Vessel 651** | 822 – 957 | 192 | 631.0 | ±14.9 | 0.1202 | 0.0587 | 3 |

### Double Q-Learning — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3002 | 38458 | 116.5 | ±19.4 | 1.0107 | 2.4287 | 434 |
| **Vessel 563** | 751 – 1842 | 800 | 794.5 | ±14.7 | 0.1380 | 0.0686 | 10 |
| **Vessel 576** | 726 – 1783 | 460 | 553.8 | ±21.2 | 0.0678 | 0.0572 | 4 |
| **Vessel 533** | 709 – 1782 | 399 | 688.7 | ±21.9 | 0.1084 | 0.0569 | 5 |
| **Vessel 3547** | 2915 – 3002 | 222 | 84.0 | ±13.0 | 0.4103 | 0.0723 | 3 |
| **Vessel 1699** | 1628 – 1801 | 191 | 838.0 | ±35.3 | 0.0211 | 0.0081 | 3 |

### Linear FA (Tile Coding) — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3002 | 2987 | 149.8 | ±176.8 | 1.4505 | 0.0000 | 1 |

### Dyna-Q — 5 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 3002 | 40505 | 112.0 | ±18.5 | 1.0719 | 2.4582 | 468 |
| **Vessel 15** | 93 – 1875 | 674 | 674.5 | ±25.8 | 0.0629 | 0.1191 | 9 |
| **Vessel 39** | 156 – 1801 | 665 | 761.2 | ±21.0 | 0.1298 | 0.1053 | 8 |
| **Vessel 14** | 91 – 1748 | 243 | 577.0 | ±28.0 | 0.1071 | 0.0986 | 4 |
| **Vessel 726** | 884 – 966 | 187 | 765.3 | ±16.6 | 0.1261 | 0.0189 | 3 |

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

![RL Agent Comparison — Croatia 2507_2](rl_comparison_croatia_2507_2.png)
