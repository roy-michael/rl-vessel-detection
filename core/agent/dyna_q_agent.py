import random
import numpy as np
from core.agent.base_agent import TDBaseAgent

class DynaQAgent(TDBaseAgent):
    """
    Dyna-Q Agent (Lesson 7 — Planning & Models).

    Extends Q-Learning by maintaining a deterministic environment model
    Model[(s, a)] -> (next_s, reward). After each real experience, the agent
    performs n_planning simulated planning updates drawn uniformly from the
    previously observed state-action pairs. This greatly improves sample
    efficiency because the agent effectively replays and learns from past
    transitions multiple times per real step.

    Update sequence per real step:
      1. Real interaction:  Q(s, a) <- Q-learning update
      2. Model update:      Model[(s, a)] <- (s', r)
      3. Planning loop (n_planning times):
           Pick (s_i, a_i) uniformly from model keys
           (s', r) <- Model[(s_i, a_i)]
           Q(s_i, a_i) <- Q-learning update
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.15, n_planning=20):
        super().__init__(alpha, gamma, epsilon)
        self.n_planning = n_planning
        self.model = {}          # {(state, action): (next_state, reward)}
        self._model_keys = []    # Ordered list for O(1) random sampling

    def learn(self, state, action, reward, next_state):
        # --- Step 1: Real Q-learning update ---
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_q = self.q_table[state][action]
        max_future_q = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * max_future_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)

        # --- Step 2: Model update ---
        key = (state, action)
        if key not in self.model:
            self._model_keys.append(key)
        self.model[key] = (next_state, reward)

        # --- Step 3: Planning loop ---
        if len(self._model_keys) == 0:
            return
        n = min(self.n_planning, len(self._model_keys))
        sample_keys = random.choices(self._model_keys, k=n)
        for sim_state, sim_action in sample_keys:
            sim_next_state, sim_reward = self.model[(sim_state, sim_action)]
            self._ensure_state(sim_state)
            self._ensure_state(sim_next_state)
            sim_old_q = self.q_table[sim_state][sim_action]
            sim_max_q = float(np.max(self.q_table[sim_next_state]))
            sim_target = sim_reward + self.gamma * sim_max_q
            self.q_table[sim_state][sim_action] = sim_old_q + self.alpha * (sim_target - sim_old_q)
