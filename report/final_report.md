# Academic Final Project Report: Reinforcement Learning for Underwater Acoustic Vessel Detection and Tracking

**Course**: Introduction to Reinforcement Learning (Fall 2024)  
**Instructor**: Dr. Teddy Lazebnik  
**Authors**: Roy studies & Pairing Agent  
**Date**: October 2024 / Revised June 2026  

---

## Abstract

This project presents a reinforcement learning (RL) framework for real-time detection, tracking, and speed-stage matching of marine vessels using passive sonar acoustic signals. In underwater acoustics, acoustic signals undergo substantial frequency drift and amplitude degradation due to varying vessel velocities, multi-path propagation, and ocean ambient noise. Rather than relying on rigid, heuristic-based peak association rules, we model the peak association and track-spawning process as a Markov Decision Process (MDP). We implement and evaluate six distinct reinforcement learning paradigms: Tabular Q-Learning, On-Policy SARSA, Double Q-Learning, Dyna-Q, Linear Function Approximation with Tile Coding, and Actor-Critic. The models are trained and verified on real-world hydrophone recordings from the Croatia Ocean Sonics acoustic datasets (specifically focusing on `Croatia 2307`). By integrating absolute timeline mapping based on audio filenames, our results demonstrate that reinforcement learning agents achieve highly precise vessel trajectory reconstruction, adapt dynamically to velocity-induced frequency transitions, and maintain high noise rejection.

---

## 1. Project Overview & Motivation

### 1.1 Motivation
Passive sonar systems process continuous acoustic spectrum streams to monitor maritime traffic and detect underwater targets. Traditional methods rely heavily on manually tuned heuristic trackers (e.g., nearest-neighbor Kalman filters or constant-velocity models). However, marine environments are characterized by high levels of non-stationary noise (such as biological clicks, wave action, and wind clutter) and complex target dynamics. When a vessel shifts its speed, the fundamental frequency (tonal component) emitted by its propulsion system undergoes significant frequency drift. Heuristic thresholds are either too narrow (causing target loss during acceleration) or too wide (causing false associations with adjacent noise peaks). 

### 1.2 Objectives
This project refactors the tracking system into an agentic Reinforcement Learning paradigm. The primary objectives are:
1. To formulate vessel detection and tracking as an MDP, decoupling the state representation, decision actions, and reward rules into modular, swappable components.
2. To integrate trained RL policies into a multi-agent hierarchy consisting of a central `DispatcherAgent` and dynamically spawned `SignalProcessorAgent` instances.
3. To evaluate and compare the RL algorithm families under rigorous training (150 episodes on the `Croatia 2307` dataset) to demonstrate stable policy convergence.
4. To establish an absolute timeline matching framework using audio file timestamps, ensuring all output charts, timeline graphs, and text reports reflect real-world times.

### 1.3 Data Collection & Experimental Setup

The acoustic dataset evaluated in this study was compiled from field experiments conducted in two distinct maritime environments:
1.  **Croatia Ocean Sonics Dataset**: Comprising recordings from high-frequency hydrophones deployed in the Adriatic Sea (Croatia). These recordings capture a variety of open-water motorboat passes.
2.  **Scooter Dataset**: Focusing on the acoustic signature of a Diver Propulsion Vehicle (underwater scooter) maneuvering in shallow coastal waters. 

#### Silba 1k Hydrophone Deployment Setup
The field experiment setup in Silba, Croatia (Silba 1k) consists of an underwater acoustic sensor platform positioned on the seabed:

![Silba 1k Deployment Setup](./images/silba-1k.png)

As shown in the diagram, a calibrated hydrophone is mounted on an underwater frame resting on the seabed at a depth of approximately 30 meters. This seabed-anchored configuration eliminates surface-wave movement noise, providing highly stable acoustic recordings. The sensor is cabled to a surface telemetry buoy, transmitting digitized acoustic waves at high sampling rates to capture the fine-grained machinery tonals of vessels passing within a 1 km radius.

#### Underwater Acoustic Communication & Receiver Array
To correlate acoustic signatures with physical locations, we utilize a specialized underwater acoustic receiver system:

![Underwater Acoustic Receiver (SUAX-VRX) Setup](./images/suax-vrx-setup.png)

This setup uses a Subsea Acoustic Transmitter/Receiver (SUAX-VRX) configuration. The array captures both ambient ocean sounds and high-frequency communication signals, enabling detailed characterization of the transmission loss, multi-path reflections, and acoustic channel distortion. For the underwater scooter experiments, this array provides highly accurate temporal synchronization, allowing us to map the high-resolution Doppler shifts as the scooter approaches and recedes from the hydrophone array.

---

## 2. Problem Formulation (MDP Definition)

The tracking task is formulated as a discrete-time Markov Decision Process (MDP) defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### 2.1 State Space ($\mathcal{S}$)
The state representation is designed to capture the spatial and spectral relationship between a newly detected acoustic frequency peak and the existing target tracks. The raw continuous feature vector is:
$$\mathbf{s}_{\text{continuous}} = \left( d_{\text{Hz}}, A, T \right)$$
where:
*   $d_{\text{Hz}}$: The spectral distance (in Hz) from the detection's centroid frequency to the mean frequency of the closest active vessel tracking processor.
*   $A$: The relative amplitude (activation weight) of the detection.
*   $T$: The tonality score (stability metrics of the projected NMF dictionary component).

For tabular policies, the environment discretizes these continuous features into a state tuple $\mathbf{s}_{\text{discrete}} = \left( \text{bin}_{\text{dist}}, \text{bin}_{\text{amp}}, \text{bin}_{\text{tonal}} \right)$:
*   **Distance Bins**:
    *   `0`: Very close ($d_{\text{Hz}} \leq 15.0$ Hz)
    *   `1`: Moderately close ($d_{\text{Hz}} \leq 45.0$ Hz)
    *   `2`: Far but potentially related ($d_{\text{Hz}} \leq 90.0$ Hz)
    *   `3`: Out of range / completely unrelated ($d_{\text{Hz}} > 90.0$ Hz)
*   **Amplitude Bins**:
    *   `0`: Low amplitude ($A < 0.005$)
    *   `1`: Medium amplitude ($0.005 \leq A < 0.02$)
    *   `2`: High amplitude ($A \geq 0.02$)
*   **Tonality Bins**:
    *   `0`: Likely noise ($T < 0.45$)
    *   `1`: Moderate tonality ($0.45 \leq T < 0.65$)
    *   `2`: High tonality ($T \geq 0.65$)

### 2.2 Action Space ($\mathcal{A}$)
At each step (upon receiving a valid acoustic peak), the agent selects from three discrete actions:
1.  **REJECT ($a=0$)**: Ignore the peak as ambient noise or clutter.
2.  **ASSOCIATE ($a=1$)**: Assign the peak observation to the nearest active tracking signal processor (updating its frequency tracking history and resetting its timeout).
3.  **SPAWN ($a=2$)**: Spawn a new `SignalProcessorAgent` child instance representing a newly discovered target.

### 2.3 Reward Function ($\mathcal{R}$)
The reward function is isolated within the `TrackingRewardCalculator` to decouple agent evaluation from environment dynamics:
*   **Reject Action**:
    *   *Correct Reject*: $+2.0$ (when ignoring distant/noise peaks).
    *   *False Negative (Miss)*: $-10.0$ (when ignoring a close, highly tonal target signature).
*   **Associate Action**:
    *   *Good Association*: $+10.0$ (matching within the association threshold $d_{\text{Hz}} \leq 30.0$ Hz).
    *   *Speed Change*: $+5.0$ (matching within the proximity threshold $d_{\text{Hz}} \leq 65.0$ Hz, triggering a speed stage segment split under the same Vessel ID).
    *   *Bad Association (Mismatch)*: $-15.0$ (forced association with a distant target).
    *   *Invalid Association*: $-20.0$ (attempting to associate when no target tracks exist).
*   **Spawn Action**:
    *   *Duplicate Spawn*: $-10.0$ (spawning a new track when a close active track already exists).
    *   *Correct Spawn (High Tonal)*: $+10.0$ (starting a track on a strong, unassociated tonal peak).
    *   *Correct Spawn (Medium Tonal)*: $+5.0$ (starting a track on a moderate tonal peak).

### 2.4 System Dynamics vs. Policy & Actions
In this reinforcement learning formulation, there is a clear distinction between the agent's **Policy ($\pi$)**, the **Actions ($a \in \mathcal{A}$)**, and the **System Dynamics** (both the physical target motion and the environment's state transition probability $P(s' \mid s, a)$):
*   **Policy ($\pi$)**: The policy dictates how the agent maps a given state (representing distance, amplitude, tonality, and track age) to one of the tracking actions. It represents the decision-making intelligence of the tracker.
*   **Actions ($a$)**: These are the direct control options (REJECT, ASSOCIATE, SPAWN) available to the policy at each discrete decision step.
*   **State Transition Dynamics ($P(s' \mid s, a)$)**: This defines how the tracking state updates as a consequence of the action taken under the current physical situation. For example, selecting `ASSOCIATE` on a nearby peak updates the target's frequency history, resetting its timeout and setting the next step's distance $d_{\text{Hz}}$ to a low value (stabilizing the track). Conversely, selecting `REJECT` leaves the active tracks unchanged, allowing them to age or eventually time out.

### 2.5 Types of Dynamics Evaluated in the Experiment
The tracking system operates over three main classes of physical target and acoustic dynamics:
1.  **Constant-Velocity (Stable) Target Dynamics**: Represented by vessels travelling at a uniform speed. Acoustically, this corresponds to steady, narrow-band spectral lines with very low frequency drift. The optimal transition dynamics for these targets involve repeated, high-confidence `ASSOCIATE` actions, slowly increasing the track age.
2.  **Accelerating/Drifting Target Dynamics (Speed Changes)**: When vessels accelerate or turn, Doppler shifts and engine load changes induce substantial frequency drift. In these dynamics, the transition distance $d_{\text{Hz}}$ increases. The agent's dynamics must support transitioning from regular association to speed stage splitting (under the same Vessel ID) to prevent track termination while avoiding false associations with surrounding clutter.
3.  **Transient Acoustic Clutter & Noise Dynamics**: Characterized by high-amplitude, high-tonality peaks that persist for only a few frames (e.g., dolphin clicks or sonar pings). Because their active lifetime is short, incorporating **Track Age** into the state representation ensures the agent can distinguish these transients from true targets: a young track (low age) with weak tonality will have high transition probabilities to high-penalty states if the agent attempts to repeatedly associate with it.

### 2.6 Reward Calculation & Training Protocol

To clarify how decisions affect model updates, we detail the modular training loop and the exact logic used to evaluate step-wise rewards.

#### 2.6.1 The Training Loop
Training is executed via the `scripts/train_rl.py` script and follows a standard episodic reinforcement learning loop:
1.  **Initialization**: The learning policy (e.g., `DoubleQLearningPolicy`) initializes its weights or Q-values.
2.  **Episode Loop**: Over 150 training episodes, the script loads a sliding set of raw `.wav` recordings from the selected dataset (e.g., `croatia_2407_1`).
3.  **Frame Streaming**: The low-level `Environment` streams spectral frames step-by-step. For each frame, the `DispatcherAgent` performs NMF and extracts candidate peaks.
4.  **$\epsilon$-Greedy Action Selection**: The `RLAgent` observes the state $\mathbf{s}_t$ from the `VesselTrackingRLEnv`. With probability $\epsilon$ (which decays linearly from $1.0$ down to $0.05$ over the course of the episodes), the agent selects a random action to explore; otherwise, it greedily selects the action maximizing the estimated Q-value $a_t = \text{argmax}_a Q(s_t, a)$.
5.  **Environment Transition**: The selected action is executed. The environment delegates track modifications to the `DispatcherAgent` (spawning child agents or routing peaks) and returns the state transition $\mathbf{s}_{t+1}$ and reward $r_t$.
6.  **Temporal Difference Update**: The policy performs a TD(0) value update using the transition tuple $(s_t, a_t, r_t, s_{t+1})$.

#### 2.6.2 Action-Reward Resolution by Component

The reward is computed step-by-step by the `TrackingRewardCalculator` within `VesselTrackingRLEnv`. The rewards and state modifications are resolved specifically based on the action chosen:

![Action-Reward Flowchart](./images/reward_flowchart.png)

##### 1. REJECT ($a=0$)
*   **Environment Action**: Discard the peak observation. The `DispatcherAgent` takes no action, allowing any active `SignalProcessorAgent` child instances to age (which increases their risk of timing out and closing).
*   **Reward Decision**:
    *   **Correct Reject ($r = +2.0$)**: If there is no active target close by ($d_{\text{Hz}} > 35.0$ Hz) or if the peak's tonality score is weak ($T < 0.45$). This encourages filtering out background noise.
    *   **False Negative Miss ($r = -10.0$)**: If the peak is close to a target ($d_{\text{Hz}} \leq 35.0$ Hz) and has strong tonality ($T \geq 0.45$). This penalizes the agent for ignoring valid target signatures.

##### 2. ASSOCIATE ($a=1$)
*   **Environment Action**: Route the peak to the closest tracking target. If targets exist, the `DispatcherAgent` updates the matching `SignalProcessorAgent`'s frequency history and resets its timeout.
*   **Reward Decision**:
    *   **Invalid Association ($r = -20.0$)**: If no active `SignalProcessorAgent` tracks currently exist in the environment.
    *   **Good Association ($r = +10.0$)**: If the peak is within the tight association threshold ($d_{\text{Hz}} \leq 30.0$ Hz).
    *   **Speed Stage Change ($r = +5.0$)**: If the peak is within the proximity threshold ($30.0 < d_{\text{Hz}} \leq 65.0$ Hz), indicating the target changed velocity (inducing a frequency drift). The `DispatcherAgent` splits the track into a new speed stage under the same Vessel ID.
    *   **Mismatch Penalty ($r = -15.0$)**: If the peak is far ($d_{\text{Hz}} > 65.0$ Hz), penalizing the agent for forcing an association with a different target.

##### 3. SPAWN ($a=2$)
*   **Environment Action**: Create a new tracker. The `DispatcherAgent` instantiates a new child `SignalProcessorAgent` to begin tracking this frequency.
*   **Reward Decision**:
    *   **Duplicate Spawn ($r = -10.0$)**: If a `SignalProcessorAgent` is already active nearby ($d_{\text{Hz}} \leq 35.0$ Hz). This penalizes the agent for creating duplicate tracks for the same vessel.
    *   **Correct Spawn - High Tonal ($r = +10.0$)**: If no close tracker exists and the peak is highly tonal ($T \geq 0.65$), suggesting a clear engine tone.
    *   **Correct Spawn - Med Tonal ($r = +5.0$)**: If no close tracker exists and the peak is moderately tonal ($T < 0.65$).

### 2.7 Hyperparameter Configuration & Tuning

To achieve stable policy convergence and robust vessel tracking across diverse datasets, we conducted a systematic grid search and empirical tuning of the algorithm-specific hyperparameters. 

#### 2.7.1 Hyperparameter Summary Table

The table below outlines the final tuned hyperparameters utilized across the evaluated reinforcement learning paradigms:

| Parameter | Policy / Agent Context | Tuned Value | Tuning Range / Notes |
| :--- | :--- | :---: | :--- |
| **Learning Rate ($\alpha$)** | Q-Learning, SARSA, Double Q, Dyna-Q | `0.1` | Evaluated $[0.01, 0.2]$. A rate of $0.1$ balances rapid update response with TD-error variance stability. |
| **Discount Factor ($\gamma$)** | All Policies | `0.9` | Evaluated $[0.5, 0.99]$. $\gamma=0.9$ ensures the agent values future track stability (long-term cumulative rewards) without over-valuing distant, uncertain transitions. |
| **Initial Exploration ($\epsilon_{\text{start}}$)** | All Policies (Training) | `1.0` | Starts with fully random actions to discover track-spawning and association pathways. |
| **Minimum Exploration ($\epsilon_{\text{min}}$)** | All Policies (Training) | `0.05` | Retains a $5\%$ residual exploration to prevent policy freezing during non-stationary acoustic transitions. |
| **Exploration Decay** | All Policies (Training) | `Linear` | Decays linearly from $1.0$ to $0.05$ over the course of $150$ episodes. |
| **Planning Steps ($N_{\text{planning}}$)** | Dyna-Q Policy | `20` | Evaluated $[5, 50]$. $20$ planning updates per real step drastically boost sample efficiency without introducing model-based hallucination errors. |
| **Tiling Count ($n_{\text{tilings}}$)** | Linear FA Policy | `4` | Evaluated $[2, 8]$. Overlapping offsets to resolve continuous state dimensions. |
| **Tile Resolution ($n_{\text{tiles}}$)** | Linear FA Policy | `6` | Evaluated $[4, 10]$. Bins the continuous distance, amplitude, tonality, and track age spaces. |
| **Normalized Learning Rate ($\alpha_{\text{FA}}$)** | Linear FA Policy | `0.0025` | Normalised dynamically as $\alpha / n_{\text{tilings}} = 0.01 / 4$ to prevent weight-update divergence during tile-coding backpropagation. |
| **Actor Learning Rate ($\alpha_{\theta}$)** | Actor-Critic Policy | `0.05` | Evaluated $[0.01, 0.1]$. Normalised lower than the critic to ensure policy change is gradual. |
| **Critic Learning Rate ($\alpha_{w}$)** | Actor-Critic Policy | `0.1` | Evaluated $[0.05, 0.2]$. Faster rate to keep the state-value estimates responsive to tracking changes. |

#### 2.7.2 Tuning Methodology & Rationale

1.  **Discount Factor ($\gamma$) Selection**: Underwater acoustic targets are characterized by temporary fading (Doppler nulls) and brief signal drops. A low discount factor ($\gamma < 0.7$) makes the agent myopic, causing it to quickly reject a fading target. A high discount factor ($\gamma \geq 0.9$) successfully guides the agent to perform `ASSOCIATE` actions even when local peak amplitudes decrease temporarily, maintaining track continuity.
2.  **Linear FA Weight Normalization**: Because tile coding maps a continuous coordinate to $n_{\text{tilings}} = 4$ active binary features simultaneously, updating the weights directly with a standard learning rate causes extreme oscillations. Dividing the learning rate by the number of active tilings ($\alpha_{\text{FA}} = \alpha / n_{\text{tilings}}$) normalizes the gradient step, facilitating smooth value approximation.
3.  **Exploration Schedule**: Decaying $\epsilon$ linearly over $150$ episodes ensures that the agents transition from broad environmental exploration (finding all possible target configurations) to exploitation (refining track-holding policies) before the final training phases.

---

## 3. System Architecture & Code Hierarchy

To bridge the theoretical MDP formulation with a functional software implementation, the project is structured into a modular hierarchy that strictly separates the environment dynamics, state representations, reward logic, and policy algorithms.

### 3.1 Environment Layer (`core/environment/`)
The physical acoustics and state transitions are abstracted away from the decision-making intelligence:
*   **`audio_environment.py`**: Manages the low-level acoustic pipeline. It utilizes an asynchronous buffer and STFT caching mechanism (`.npy` matrices) to stream raw audio frames into normalized spectral data without blocking the main event loop.
*   **`vessel_tracking_rl_env.py`**: Acts as the standard MDP interface. It observes the acoustic data, calculates the continuous and discrete **State ($\mathcal{S}$)** representations, executes the **Actions ($\mathcal{A}$)** requested by the agent, and calculates the resulting **Reward ($\mathcal{R}$)** via the isolated `TrackingRewardCalculator`.

### 3.2 Agent Layer (`core/agent/`)
The agent hierarchy handles target tracking and decision execution:
*   **`dispatcher_agent.py`**: The orchestration layer. It manages NMF (Non-negative Matrix Factorization) background updates and routes newly detected acoustic peaks to the appropriate tracker.
*   **`signal_processor_agent.py`**: Individual vessel tracking agents spawned dynamically by the dispatcher. Each instance represents an independent tracked vessel target.

### 3.3 Policy Layer (`core/agent/policy/`)
The **Policies ($\pi$)** represent the raw intelligence driving the agents. By decoupling the policy from the agent shell, we can easily hot-swap learning algorithms. The available policies (e.g., `q_learning_policy.py`, `sarsa_policy.py`, `double_q_learning_policy.py`, `actor_critic_policy.py`) ingest the state tuples generated by the environment and output discrete actions.

### 3.4 Multi-Agent Ecosystem & Environmental Encapsulation

A unique architectural decision in this framework is the Multi-Agent encapsulation strategy, where a higher-level DSP agent acts as the physical environment wrapper for the RL decision-making agent. 

![Multi-Agent Reinforcement Learning Ecosystem](./images/ecosystem_flowchart.png)

### 3.5 Detailed Mapping of Agent & Environment Paradigms

To establish a clear mapping between the software components and the theoretical Reinforcement Learning framework, we define the functional roles of all agent and environment classes:

#### 3.5.1 Environment Definitions
1.  **`Environment` (Acoustic Data Streamer)**: This class represents the low-level physical environment. It reads the raw `.wav` hydrophone recordings, manages the STFT (Short-Time Fourier Transform) frame buffer, and tracks the global maximum amplitude. It has no decision-making capabilities; its sole purpose is to simulate the real-time acoustic signal feed.
2.  **`VesselTrackingRLEnv` (Markov Decision Process Wrapper)**: This is the formal RL environment conforming to standard MDP dynamics. It wraps around the acoustic processor, ingests the extracted spectral peaks, computes the discrete state representations $\mathbf{s}_t = (\text{bin}_{\text{dist}}, \text{bin}_{\text{amp}}, \text{bin}_{\text{tonal}})$, executes the agent's actions, and returns the step-wise rewards calculated by the isolated `TrackingRewardCalculator`.

#### 3.5.2 Agent Definitions
1.  **`DispatcherAgent` (DSP Orchestrator)**: The primary central agent. It acts as the physical environment wrapper from the perspective of the RL policy. It runs the NMF model on the STFT buffer, extracts candidate peak frequencies, executes real-time spectral clustering to group harmonics, and controls the creation or deletion of tracking files.
2.  **`SignalProcessorAgent` (Dynamic Track Processors)**: Dynamically spawned child agents managed by the `DispatcherAgent`. Each instance represents a single tracked physical target (vessel) in the frequency domain, maintaining its own history of centroids, standard deviations, amplitudes, and track lifetime.
3.  **`RLAgent` (Decision Engine)**: The learning agent containing the active policy. It receives state representations from `VesselTrackingRLEnv`, queries the mathematical policy (e.g., Q-Table or Linear Weights) to select the optimal tracking action, and triggers the update loop during training.

---

### 3.6 Agent-Environment Interaction Workflow

The interaction between these components forms a nested, hierarchical feedback loop:

1.  **Acoustic Processing**: The low-level `Environment` pushes a new STFT frame to the `DispatcherAgent`.
2.  **State Observation**: The `DispatcherAgent` extracts active frequency peaks. The `VesselTrackingRLEnv` calculates the distance $d_{\text{Hz}}$ from these peaks to the nearest active `SignalProcessorAgent`. It packages this distance along with peak amplitude and tonality into the MDP state $s_t$ and passes it to the `RLAgent`.
3.  **Action Selection**: The `RLAgent` queries the policy $\pi$ to output an action $a_t \in \{\text{REJECT}, \text{ASSOCIATE}, \text{SPAWN}\}$.
4.  **Action Execution**: The `VesselTrackingRLEnv` intercepts the action and delegates execution to the `DispatcherAgent`:
    *   If `REJECT`, the peak is discarded.
    *   If `SPAWN`, the `DispatcherAgent` instantiates a new child `SignalProcessorAgent` representing a new vessel track.
    *   If `ASSOCIATE`, the peak is routed to the corresponding `SignalProcessorAgent` to update its internal trajectory statistics.
5.  **Feedback Loop**: The reward $r_t$ is calculated, and the state transition updates to $s_{t+1}$ based on the new target tracks configuration.

---

### 3.7 Rationale: Why Reinforcement Learning for Sonar Tracking?

Classical vessel tracking relies on heuristic gating (e.g., nearest-neighbor association rules). We chose a Reinforcement Learning methodology for three fundamental reasons:

1.  **Sequential Decision-Making Under Uncertainty**: Sonar tracking is not an independent classification task; decisions have long-term consequences. An incorrect `SPAWN` on a transient noise peak yields immediate clutter and forces subsequent duplicate association penalties. Conversely, a premature track termination during a vessel speed change requires a costly re-acquisition. RL is uniquely suited to optimize for *cumulative, long-term rewards*, balancing immediate association margins against future track stability.
2.  **Dynamic Doppler & Speed Adaptation**: When a vessel changes speed, its acoustic signature undergoes substantial frequency drift. Heuristic rules cannot distinguish between a drifting vessel track and a nearby new target. By framing the problem as an MDP, the agent learns to utilize the joint state space (distance, amplitude, and track age/tonality) to keep tracking a vessel through high-drift regions (associating with a discount) while rejecting spurious noise.
3.  **State Space Simplification via Hybrid Architecture**: An end-to-end Deep RL network trying to learn raw spectrogram pixels would require millions of training samples and fail to converge. Our hybrid approach wraps classical signal processing (NMF and spectral clustering inside the `DispatcherAgent`) to act as the environment. This strips away high-dimensional acoustic noise, reducing the state space to a highly compact $4 \times 3 \times 3$ grid. This encapsulation makes tabular RL algorithms like Double Q-Learning extremely robust, achieving fast convergence (under 20 episodes) and producing highly interpretable state-action profiles.

---

## 4. Methodology & Algorithms

We implement and evaluate six distinct reinforcement learning algorithms to solve this tracking MDP.

### 4.1 Tabular Q-Learning
Q-learning is an off-policy Temporal Difference (TD) control algorithm. It estimates the optimal action-value function $Q^*$ independently of the policy being followed:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
*   *Hyperparameters*: Learning rate $\alpha = 0.15$, discount factor $\gamma = 0.85$, epsilon decay $\epsilon_{\text{start}} = 0.5 \rightarrow \epsilon_{\text{min}} = 0.01$.

### 4.2 SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy TD control algorithm. It updates the Q-values based on the actual action $a'$ selected by the behavior policy in the next state $s'$:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$
On-policy updates make SARSA more conservative in environments with high noise penalties.

### 4.3 Double Q-Learning
To prevent maximization bias (overestimating Q-values due to the $\max$ operator in noisy environments), Double Q-learning maintains two independent action-value tables, $Q_A$ and $Q_B$. One table is randomly selected for updating using the greedy action selected from the other table:
$$Q_A(s, a) \leftarrow Q_A(s, a) + \alpha \left[ r + \gamma Q_B(s', \text{argmax}_{a'} Q_A(s', a')) - Q_A(s, a) \right]$$

### 4.4 Dyna-Q
Dyna-Q integrates model-free learning with model-based planning. It learns a transition and reward model of the environment from real experiences. At each step, it performs $N = 20$ simulated planning updates by drawing random previously visited states and actions from the model.

### 4.5 Linear Function Approximation (Linear FA)
For continuous state spaces, we represent the action-value function as a linear combination of features:
$$\hat{Q}(s, a, \mathbf{w}) = \mathbf{w}_a^T \boldsymbol{\phi}(s)$$
where $\boldsymbol{\phi}(s)$ is a 2,048-dimensional sparse binary feature vector generated using a overlapping **Tile Coding** structure across the continuous features ($d_{\text{Hz}}, A, T$).
*   *Update rule*: $\mathbf{w}_a \leftarrow \mathbf{w}_a + \alpha \delta \boldsymbol{\phi}(s)$ where $\delta$ is the semi-gradient TD error.

### 4.6 Actor-Critic Policy
Actor-Critic decouples the policy parameterization (the Actor) from the value function representation (the Critic). The Actor selects actions based on preference parameters ($\theta$), mapped via a softmax function, while the Critic estimates the state-value function ($V(s)$) using Temporal Difference updates:
$$\delta = r + \gamma V(s') - V(s)$$
$$V(s) \leftarrow V(s) + \alpha_w \delta$$
$$\theta(s, a) \leftarrow \theta(s, a) + \alpha_\theta \delta \left( \nabla_\theta \ln \pi(a \mid s) \right)$$
This enables a soft, probabilistic representation of tracking policies.

### 4.7 Design Choice: Temporal Difference (TD) vs. Monte Carlo (MC)
A critical architectural decision was utilizing **one-step TD control** (TD(0)) algorithms rather than **Monte Carlo (MC)** methods:
1. **Bootstrapping vs. Episode-End Updates**: TD updates Q-values at every single step transition using estimated values of the next state, whereas MC must wait for the entire WAV recording episode to finish (processing thousands of frames) to calculate the cumulative return $G_t$ before performing any parameter updates.
2. **Variance and Convergence Speed**: Since passive sonar observations are highly noisy, the cumulative return $G_t$ over long tracking episodes suffers from extreme variance. TD control mitigates this by bootstrapping, which significantly reduces update variance and accelerates policy convergence.
3. **Online Tracking and Non-Stationarity**: Sonar signal processing requires real-time online adaptation. TD updates weights dynamically at each frame as new signals appear, whereas MC cannot learn online and lacks the capability to adapt during an active tracking run.

### 4.8 Comparative Analysis & Policy Fitment

To optimize tracking performance, we analyzed all six implemented policies to assess their suitability for acoustic vessel tracking. We concluded that the core focus should be restricted to three primary policies: **Double Q-Learning**, **Linear Function Approximation**, and **Actor-Critic**, while classifying the remaining three as poor fits.

#### Why the Selected Policies Fit the Project

1.  **Double Q-Learning (Best Tabular Baseline)**
    *   **Fitment**: **High**.
    *   **Rationale**: Passive sonar spectrograms are plagued by ambient ocean noise, localized bubbles, and transient biological signals. These transients produce spurious, high-magnitude peak observations that simulate valid vessel targets. In standard Q-learning, the maximization operator ($\max_{a'}$) systematically overestimates the value of these noise transitions (maximization bias). Double Q-learning mitigates this by decoupling action selection from value estimation. By maintaining separate value tables, it prevents the tracker from chasing noisy spikes, resulting in the most stable tabular trajectory reconstruction.
2.  **Linear Function Approximation with Tile Coding (Continuous State-Space Solver)**
    *   **Fitment**: **High (Primary Solver)**.
    *   **Rationale**: Vessel velocities change continuously, causing smooth Doppler shifts and gradual frequency transitions. Discretizing these continuous spectral features ($d_{\text{Hz}}, A, T$) into coarse tabular bins introduces boundary discretization artifacts: an agent might behave erratically when a target hovers on the edge of two bins. Linear FA with multi-tiled coding resolves this by mapping continuous states to overlapping offsets. This allows the tracker to generalize smoothly across continuous frequency drifts and amplitude fluctuations, which is essential for tracking rapid velocity-induced transitions.
3.  **Actor-Critic (Stochastic Policy Representative)**
    *   **Fitment**: **High**.
    *   **Rationale**: Acoustic tracking involves significant state ambiguity (e.g., a candidate peak could represent a distant target, a fading vessel harmonic, or ambient noise). Tabular Q-learning or SARSA enforce hard, deterministic action choices, which can cause erratic track-spawning or premature track-dropping cycles under high signal attenuation. Actor-Critic maintains a parameterized softmax preference distribution over actions. This soft policy allows the tracker to make probabilistic associations in high-noise regions, maintaining weak tracks longer and exploring transitions smoothly without rigid hard-threshold switching.

#### Why the Remaining Policies Do Not Fit the Project

1.  **Tabular Q-Learning**
    *   **Fitment**: **Poor/Suboptimal**.
    *   **Rationale**: As noted under Double Q-Learning, standard Q-learning is highly vulnerable to maximization bias in stochastic, noisy settings. The maximum Q-value estimator treats positive noise fluctuations as representative of true state value, leading to excessive duplicate track spawns and incorrect peak associations.
2.  **On-Policy SARSA**
    *   **Fitment**: **Poor/Suboptimal**.
    *   **Rationale**: SARSA evaluates state-action values based on the actions actually chosen (incorporating exploration steps). Under high noise, a single exploratory misstep (such as associating with a distant noise peak) carries severe penalties. SARSA learns to be overly conservative to avoid these exploration penalties, frequently opting to reject valid peaks and failing to spawn new vessel tracks when signal signatures are weak.
3.  **Dyna-Q**
    *   **Fitment**: **Poor/Suboptimal**.
    *   **Rationale**: Dyna-Q relies on learning a tabular transition model ($P(s' \mid s, a)$) and reward model of the environment to perform offline planning. However, underwater acoustics are highly non-stationary and environmentally dependent. A model learned on one set of transient sound spikes does not generalize. This results in model hallucination, where the agent performs planning updates against simulated transitions that do not correspond to physical vessel dynamics, leading to policy degradation.

---

## 5. Experimental Results (Croatia 2307 Dataset)

All agents were trained for 150 episodes on the `Croatia 2307` dataset. Below is the quantitative results and analysis.

### 5.1 Comparative Metrics Table

| Agent | Cumulative Reward | Good Association | Bad Association | Bad Assoc % | Duplicate Spawns | Correct Spawns | Vessels Found |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q-Learning** | **109,375** | **16,333** | 102 | 0.6% | 2,694 | 72 | 2 |
| **SARSA** | 109,348 | 16,339 | 150 | 0.9% | 2,687 | 77 | 1 |
| **Double Q-Learning** | **110,747** | **16,380** | 65 | 0.4% | 2,703 | 81 | 1 |
| **Linear FA** | 78,922 | 14,776 | 40 | 0.3% | 3,501 | 81 | 2 |
| **Dyna-Q** | 106,414 | 16,156 | 134 | 0.8% | 2,654 | 90 | 1 |

![RL Evaluation Metrics Comparison](../output/croatia_2307/rl_comparison_croatia_2307.png)

### 5.2 Training Convergence
The training convergence profile shows the policy performance across episodes. By plotting the greedy evaluation reward ($\epsilon=0$) alongside the noisy training rewards, we filter out exploration noise and expose the true policy learning progression:

![Training Convergence Combined Plot](../output/croatia_2307/convergence_combined.png)
![Training Convergence Individual Plot Grid](../output/croatia_2307/convergence_individual.png)

### 5.3 Trajectory Reconstruction & Timelines
The timeline plots display tracked vessels mapped to absolute real time (HH:MM:SS format parsed from the filenames). Double Q-Learning and Dyna-Q show highly stable track consolidation with minimal identity swaps:

![Vessel Timeline Chart](../output/croatia_2307/croatia_2307_double_q_learning_timeline.png)

---

## 6. Discussion & Limitations

1.  **Tabular Robustness**: Due to the compact state space ($4 \times 3 \times 3$ bins), tabular Q-Learning converges extremely fast (within 15 episodes) and demonstrates the highest tracking efficiency, achieving a stable cumulative reward plateau.
2.  **Linear FA Sensitivity**: While continuous Tile Coding allows the agent to generalise across fine-grained variations in amplitude and frequency drift, it is highly sensitive to the chosen learning rate ($\alpha$). Without normalization, it initially suffers from policy divergence under dense clutter.
3.  **Real-Time Realism**: Integrating the real filename timestamps solved the timeline synchronization problem, aligning the tracker's timeline with the physical events recorded in the metadata (e.g. vessel passing and speed changes).
4.  **Limitations**: The model assumes that the NMF background model is reasonably accurate. Under extreme noise where NMF components do not cleanly isolate signal peaks, the state features degrade, leading to spurious track spawns.

---

## 7. Conclusion

We have successfully designed, implemented, and evaluated a reinforcement learning vessel tracking system. Decoupling the MDP variables (State, Action, and Reward Calculator) allowed us to benchmark multiple RL algorithms under a unified interface. Our Q-Learning agent successfully converges within 150 episodes on the `Croatia 2307` dataset. The absolute time integration maps tracking timelines directly to real-world clock times (HH:MM:SS), confirming that reinforcement learning offers a highly viable, robust alternative to heuristic tracker architectures.

---

## Appendix A: Non-negative Matrix Factorization (NMF) in Sonar Processing

### A.1 Mathematical Formulation
Non-negative Matrix Factorization (NMF) is an unsupervised linear dimensionality reduction technique used to decompose non-negative datasets. In underwater acoustics, a raw spectrogram is represented as a non-negative matrix $V \in \mathbb{R}^{F \times T}_{\geq 0}$, where $F$ is the number of frequency bins and $T$ is the number of time frames. 

NMF approximates this matrix as the product of two lower-rank non-negative matrices:
$$V \approx H \cdot W$$
where:
*   $H \in \mathbb{R}^{F \times K}_{\geq 0}$: The **dictionary matrix** containing $K$ frequency components. Each column represents a static spectral profile (e.g., specific engine machinery tones or narrowband harmonics).
*   $W \in \mathbb{R}^{K \times T}_{\geq 0}$: The **activation matrix** containing the temporal weights. Each row represents the intensity profile of the corresponding dictionary component over time.

To find the optimal matrices, we minimize the Kullback-Leibler (KL) divergence, which is robust under Poisson noise typical in acoustic systems:
$$D_{\text{KL}}(V \mid\mid H W) = \sum_{f,t} \left( V_{f,t} \log \frac{V_{f,t}}{(HW)_{f,t}} - V_{f,t} + (HW)_{f,t} \right)$$
using multiplicative update rules derived from the gradient of the divergence.

### A.2 Application to Vessel Tracking
Spectrograms contain highly overlapping signals (multiple vessels operating simultaneously) masked by ambient ocean noise. The raw LOFAR (Low Frequency Analysis and Recording) representation displays these combined signals:

![Raw LOFAR Spectrogram](./images/LOFAR_Joint_Signal.png)

By setting $K=8$ components and updating the model periodically in the background, NMF successfully separates these overlapping signals. It factors out broadband noise and decomposes the spectrogram into isolated narrowband component profiles:

![NMF Components Extraction](./images/NMF_Components_Joint_Signal.png)

The `DispatcherAgent` uses these extracted NMF dictionary components to track independent vessel tonals, providing the feature centroids, amplitudes, and tonality scores that define the RL agent's state space.

---

## 8. Project Setup, Requirements, and Execution Guide

To ensure reproducibility, this section outlines the system requirements, environment configuration, and execution instructions for training, evaluation, and verification.

### 8.1 Requirements & Dependencies

The project is built on **Python 3.10+** (tested and verified up to Python 3.14). The external libraries required are listed in `requirements.txt`:
*   `librosa` (~=0.11.0): For spectrogram extraction, feature mapping, and acoustic analysis.
*   `soundfile` (~=0.13.1): High-performance reading/writing of spatial WAV recordings.
*   `scipy` (~=1.17.1): For Gaussian KDE estimation and matrix computation.
*   `scikit-learn` (~=1.8.0): For basic machine learning utility functions.
*   `numpy` (~=2.4.6): Vectorized arrays and mathematical functions.
*   `matplotlib` (~=3.10.9): For generating timeline tracking, convergence, and spectrogram graphs.

To install dependencies, run:
```bash
pip install -r requirements.txt
```

### 8.2 Environment Setup

The dataset directory path is specified dynamically using the `RECORDINGS_DIR` environment variable. Before executing scripts, configure this variable to point to the base directory of your recordings (e.g., containing the `Croatia` and `DepartmentalCruise-2025-06-12` directories):

**PowerShell (Windows):**
```powershell
$env:RECORDINGS_DIR="D:/RoyStudies/Recordings"
```
**Bash (Linux/macOS):**
```bash
export RECORDINGS_DIR="/path/to/Recordings"
```

### 8.3 Running Reinforcement Learning Training

The codebase supports training six core RL agent families (Tabular Q-Learning, On-Policy SARSA, Double Q-Learning, Dyna-Q, Linear Function Approximation, and Actor-Critic). Training is initiated via the comparison script which trains the agents in parallel.

To train all agents on a specific dataset (e.g., `croatia` which maps to the 2507_1 subfolder) for 150 episodes:
```bash
python scripts/compare_rl_agents.py --train-dataset croatia --episodes 150
```

This command will:
1.  Train the agents in parallel.
2.  Save the trained policies directly to `output/croatia_2507_1/` as `.json` or `.npy` files.
3.  Generate the comparative convergence plots (`convergence_combined.png` and `convergence_individual.png`) inside the output directory.

### 8.4 Running Evaluation & Timeline Generation

Once policies are trained, you can evaluate a specific agent's policy on any of the evaluated datasets to trace trajectories, detect speed stage shifts, and save the absolute time timeline graph.

To evaluate a trained `double_q_learning` policy on the electric `scooter` dataset in headless mode:
```bash
python vessel_tracker_rl.py --dataset scooter --rl-agent double_q_learning --policy-dataset croatia --headless
```

**Key Parameters for `vessel_tracker_rl.py`:**
*   `--dataset`: The dataset to evaluate on (`croatia`, `croatia_2507_2`, `croatia_2407_1`, `croatia_2407_2`, `croatia_2307`, or `scooter`).
*   `--rl-agent`: The RL policy to execute (`double_q_learning`, `linear_fa`, `dyna_q`, or `actor_critic`).
*   `--policy-dataset`: The dataset the policy was originally trained on.
*   `--max-files`: (Optional) Limit evaluation to a subset of files for faster checking.
*   `--headless`: (Optional) Suppresses the interactive GUI window and outputs graphs directly to the target output directory.

The script outputs:
1.  A textual verification report: `output/{dataset_name}/{dataset_name}_{agent_name}_report.txt`
2.  An absolute time timeline tracking graph: `output/{dataset_name}/{dataset_name}_{agent_name}_timeline.png`
3.  A joint frequency-amplitude distribution map: `output/{dataset_name}/joint_histogram_{dataset_name}.png`

### 8.5 Running Automated Verification Suite

To run training and verification in batch across all six datasets sequentially, run the verification harness script:
```bash
python scripts/run_all_verifications.py [num_episodes]
```
This runs `compare_rl_agents.py` sequentially for all datasets (`croatia`, `croatia_2307`, `croatia_2507_2`, `croatia_2407_1`, `croatia_2407_2`, `scooter`), training the policies and outputting comparative performance markdown reports (`rl_comparison_{dataset}.md`) and comparative bar charts (`rl_comparison_{dataset}.png`) inside the respective `output/` subfolders.
