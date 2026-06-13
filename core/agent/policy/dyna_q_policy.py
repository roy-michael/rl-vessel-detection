import random
import numpy as np
from core.agent.policy.base_policy import TabularPolicy


class DynaQPolicy(TabularPolicy):
    """
    Dyna-Q: Q-Learning augmented with a model-based planning loop.

    After each real (s, a, r, s') transition the policy:
      1. Performs a standard Q-learning update (real experience).
      2. Stores the transition in a deterministic model.
      3. Runs n_planning simulated updates drawn uniformly from the model.

    The planning loop greatly improves sample efficiency because each real
    interaction triggers multiple virtual updates.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 n_planning: int = 20):
        super().__init__(alpha, gamma)
        self.n_planning = n_planning
        self._model: dict = {}           # {(state, action): (next_state, reward)}
        self._model_keys: list = []      # For O(1) random sampling

    def update(self, state, action: int, reward: float,
               next_state, epsilon: float) -> None:
        # --- Step 1: Real Q-learning update ---
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_q = self._q_table[state][action]
        max_future_q = float(np.max(self._q_table[next_state]))
        td_target = reward + self.gamma * max_future_q
        self._q_table[state][action] = old_q + self.alpha * (td_target - old_q)

        # --- Step 2: Model update ---
        key = (state, action)
        if key not in self._model:
            self._model_keys.append(key)
        self._model[key] = (next_state, reward)

        # --- Step 3: Planning loop ---
        if not self._model_keys:
            return
        n = min(self.n_planning, len(self._model_keys))
        for sim_state, sim_action in random.choices(self._model_keys, k=n):
            sim_next_state, sim_reward = self._model[(sim_state, sim_action)]
            self._ensure_state(sim_state)
            self._ensure_state(sim_next_state)
            sim_old_q = self._q_table[sim_state][sim_action]
            sim_max_q = float(np.max(self._q_table[sim_next_state]))
            sim_target = sim_reward + self.gamma * sim_max_q
            self._q_table[sim_state][sim_action] = (
                sim_old_q + self.alpha * (sim_target - sim_old_q)
            )
