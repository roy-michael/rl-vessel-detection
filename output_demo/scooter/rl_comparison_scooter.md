# RL Agent Comparison — Scooter

| | |
| :--- | :--- |
| **Training Dataset** | `Croatia 2507_1` |
| **Evaluation Dataset** | `Scooter` |

---

## 1. Global Performance Metrics

| Agent | Cumul. Reward | Good Assoc | Bad Assoc | Bad Assoc % | Dup Spawns | Correct Spawns | Vessels Found |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q-Learning** | 303,039 | 38,138 | 52 | 0.1% | 5,872 | 315 | 6 |
| **SARSA** | 303,217 | 38,347 | 120 | 0.3% | 5,890 | 295 | 6 |
| **Double Q-Learning** | 282,811 | 37,298 | 276 | 0.7% | 4,727 | 303 | 8 |
| **Linear FA (Tile Coding)** | -476,680 | 8,926 | 37,728 | 80.9% | 0 | 0 | 1 |
| **Dyna-Q** | 282,797 | 37,331 | 75 | 0.2% | 4,871 | 281 | 4 |

---

## 2. Detected Vessel Details — Per Agent

> **Vessel ID** | **Active Window (s)** | **Duration (s)** | **Mean Freq (Hz)** | **Freq σ (Hz)** | **Mean Amp** | **Amp σ** | **Speed Stages**

### Q-Learning — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 10015 | 128472 | 497.9 | ±22.6 | 0.0232 | 0.0565 | 1068 |
| **Vessel 3386** | 4474 – 9275 | 1525 | 811.9 | ±25.9 | 0.0134 | 0.0095 | 16 |
| **Vessel 2791** | 2792 – 8238 | 527 | 1109.3 | ±23.0 | 0.0114 | 0.0043 | 7 |
| **Vessel 4441** | 8199 – 9274 | 483 | 835.9 | ±25.2 | 0.0113 | 0.0025 | 8 |
| **Vessel 683** | 1007 – 8256 | 356 | 993.6 | ±31.0 | 0.0197 | 0.0148 | 5 |
| **Vessel 6024** | 9833 – 10015 | 223 | 493.4 | ±21.3 | 0.0600 | 0.0568 | 4 |

### SARSA — 6 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 10015 | 133941 | 502.7 | ±22.8 | 0.0229 | 0.0421 | 1102 |
| **Vessel 1752** | 2250 – 9416 | 2646 | 934.4 | ±26.0 | 0.0170 | 0.0202 | 32 |
| **Vessel 5833** | 9600 – 9907 | 648 | 498.5 | ±27.6 | 0.0126 | 0.0114 | 4 |
| **Vessel 3995** | 7475 – 9122 | 519 | 1053.5 | ±23.1 | 0.0137 | 0.0050 | 7 |
| **Vessel 4411** | 8330 – 9294 | 338 | 845.3 | ±28.3 | 0.0113 | 0.0019 | 5 |
| **Vessel 3920** | 7336 – 8438 | 191 | 1220.1 | ±25.5 | 0.0113 | 0.0012 | 3 |

### Double Q-Learning — 8 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 22 – 10015 | 120093 | 509.1 | ±22.7 | 0.0208 | 0.0462 | 961 |
| **Vessel 4525** | 9459 – 10015 | 4063 | 453.3 | ±24.2 | 0.0392 | 0.0309 | 34 |
| **Vessel 1477** | 2268 – 9295 | 1508 | 833.6 | ±27.2 | 0.0135 | 0.0086 | 17 |
| **Vessel 4566** | 9490 – 9909 | 1419 | 478.0 | ±23.1 | 0.0178 | 0.0158 | 9 |
| **Vessel 4750** | 9598 – 9900 | 804 | 514.3 | ±23.3 | 0.0128 | 0.0041 | 4 |
| **Vessel 3159** | 7298 – 10015 | 597 | 843.9 | ±27.0 | 0.0113 | 0.0015 | 9 |
| **Vessel 2682** | 4346 – 8506 | 588 | 1080.0 | ±23.1 | 0.0134 | 0.0023 | 8 |
| **Vessel 3365** | 7741 – 8487 | 338 | 1158.9 | ±25.8 | 0.0128 | 0.0015 | 5 |

### Linear FA (Tile Coding) — 1 dominant vessel detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 10015 | 10001 | 527.6 | ±156.1 | 0.0317 | 0.0000 | 1 |

### Dyna-Q — 4 dominant vessels detected

| Vessel | Window (s) | Duration (s) | Mean Freq (Hz) | Freq σ (Hz) | Mean Amp | Amp σ | Stages |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vessel 1** | 15 – 10015 | 113787 | 503.7 | ±22.9 | 0.0207 | 0.0579 | 957 |
| **Vessel 3522** | 7973 – 9223 | 700 | 828.0 | ±22.7 | 0.0137 | 0.0043 | 9 |
| **Vessel 3184** | 7208 – 8698 | 455 | 1001.1 | ±23.7 | 0.0110 | 0.0021 | 6 |
| **Vessel 3568** | 8108 – 9108 | 207 | 1044.0 | ±24.3 | 0.0122 | 0.0057 | 3 |

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

![RL Agent Comparison — Scooter](rl_comparison_scooter.png)
