import numpy as np
from core.agent.policy.base_policy import TabularPolicy


class QLearningPolicy(TabularPolicy):
    """
    Off-policy TD control (Q-Learning).
    Update rule:
        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
    """

    def update(self, state, action: int, reward: float,
               next_state, epsilon: float) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self._q_table[state][action]
        max_future_q = float(np.max(self._q_table[next_state]))
        td_target = reward + self.gamma * max_future_q
        self._q_table[state][action] = old_q + self.alpha * (td_target - old_q)
