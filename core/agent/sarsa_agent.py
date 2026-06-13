from core.agent.base_agent import TDBaseAgent

class SarsaAgent(TDBaseAgent):
    """
    SARSA Agent (On-policy TD control).
    Updates Q-values using the utility of the action actually chosen in the next state:
    Q(s, a) = Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
    """
    def learn(self, state, action, reward, next_state, next_action):
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        
        # SARSA temporal difference target
        td_target = reward + self.gamma * next_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)
