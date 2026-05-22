"""
We are in the big boys (and girls, and unicorns) area now. So, the task definition is slightly more complex. We aim to optimize the operation of a smart grid system by employing multi-agent RL. Specifically, we want to: 1) Minimize energy losses, optimize resource utilization; 2) Prevent blackouts or brownouts by balancing supply and demand; 3) Effectively incorporate fluctuating renewable energy generation; 4) Optimize energy purchasing and selling for all agents; 5) Provide reliable and affordable electricity to end-users.
By modeling the smart grid as a multi-agent system, we can explore how decentralized decision-making can lead to improved overall system performance.

Agents in the smart grid system are autonomous entities with their own objectives. Each agent maintains an internal state representing its current condition and capabilities.

Power Plant Agent. State: Generation level, fuel level, maintenance status, operational costs. Actions: Discrete actions (e.g., increase generation by X%, decrease generation by X%, maintain current level) or continuous actions (generation level as a continuous value). Reward: Profit from energy sales, penalties for unmet demand, fuel costs, operational costs.

Energy Storage Agent. State: State of charge (SOC), current power flow (charging/discharging), energy price. Actions: Discrete actions (charge, discharge, idle) or continuous actions (charging/discharging rate). Reward: Profit from arbitrage, penalties for deep discharges, degradation costs.

Demand Response Agent. State: Current demand, potential demand reduction, price elasticity, customer preferences. Actions: Discrete actions (offer incentive, no incentive) or continuous actions (incentive level). Reward: Revenue from demand response programs, customer satisfaction, grid stability contributions.

The environment serves as the intermediary between agents, simulating the physical and economic aspects of the power grid. State: Aggregate demand, energy prices, system frequency, reserve margins, and other relevant grid parameters. Dynamics: Updates agent states based on actions, simulates random events (e.g., load fluctuations, equipment failures), handles energy flow and balancing. Information flow: Provides observations to agents (e.g., energy prices, system imbalance). Market clearing: Determines energy prices based on supply and demand curves.

For this task, we picked Multi-Agent Deep Deterministic Policy Gradient (MADDPG) as it handles continuous action spaces and complex interactions effectively. I choose it as generation levels, energy storage rates, and demand response levels are continuous variables, making MADDPG suitable. Moreover, the interdependent nature of power plants, energy storage, and demand response requires an algorithm that can model these interactions effectively. MADDPG addresses this by allowing agents to observe other agents' actions. In addition, it can be adapted to various reward structures and environmental dynamics, which can be useful as we do not have a clear definitions of these (the author of this blog post has a healthy self-critic).
"""
import asyncio

import librosa
import numpy as np
from sklearn.decomposition import NMF


class DispatcherAgent:
    """
    The dispatcher agent definition.
    Clear Definition: The reinforcement learning problem should be clearly defined, including:
        States: The set of all possible states the agent can be in
        Actions: The set of actions the agent can take in each state
        Rewards: The feedback signal that the agent receives for its actions
        Environment Dynamics: The rules governing the transitions between states
        Objective: The project should have a well-defined goal or objective for the agent to achieve.

    * Model Free

    """

    _state = None
    _policy = None
    _value_function = None  # ?
    _model = None  # ?
    _cumulative_reward = None

    def __init__(self, env, min_freq, max_freq, n_fft, n_components, n_max_iter):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.step = 200
        self._env = env
        self.n_fft = n_fft
        self.n_components = n_components
        self.max_iter = n_max_iter
        self.n_mels = 512
        self.hop_length = self.n_fft // 16

    async def observe(self):
        observation, reward = await self._env.observe()

        analysis = await asyncio.to_thread(self._analyze_observation, observation)

        print(len(analysis), reward)

        return None

    def _analyze_observation(self, observation):
        return [observation]

    def _calc_nmf(self, observation, n_components):
        # y, sr = librosa.load(file_path, sr=None)

        # 2. Compute the magnitude spectrogram (S)
        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))

        # Ensure max_freq doesn't exceed the Nyquist frequency
        actual_max_freq = min(self.max_freq, sr / 2.0)
        self.actual_max_freq = actual_max_freq

        # Bypass Mel and HPSS, just crop to frequency range
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        freq_mask = (freqs >= self.min_freq) & (freqs <= actual_max_freq)
        self.linear_freqs = freqs[freq_mask]

        S_input = S[freq_mask, :]

        print(f"Running sklearn NMF with k={self.n_components} (Raw={self.use_raw_linear})...")
        model = NMF(n_components=self.n_components, init='nndsvda', solver='cd', max_iter=self.max_iter,
                    random_state=42)
        W = model.fit_transform(S_input)
        H = model.components_
        errors = []

        return S, W, H, errors

    async def start(self):
        # Start background reading task
        asyncio.create_task(self._read_observations_loop())

    async def _read_observations_loop(self):
        while True:
            await self.observe()


class SignalProcessorAgent:
    pass
