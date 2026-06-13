import numpy as np
from core.agent.base_agent import TDBaseAgent

class QLearningAgent(TDBaseAgent):
    """
    Q-Learning Agent (Off-policy TD control).
    Updates Q-values using the maximum possible utility of the next state:
    Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def learn(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Q-learning temporal difference target
        td_target = reward + self.gamma * max_future_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)
