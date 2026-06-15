from core.agent.policy.base_policy import TabularPolicy


class SarsaPolicy(TabularPolicy):
    """
    On-policy TD control (SARSA).
    Update rule:
        a' ← ε-greedy(s')          # selected by the policy itself
        Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]

    SARSA's next_action is drawn internally from the same ε-greedy rule,
    so the caller does not need to know or supply it.
    """

    def update(self, state, action: int, reward: float,
               next_state, epsilon: float) -> None:
        self._ensure_state(state)
        if next_state is None:
            next_q = 0.0
        else:
            self._ensure_state(next_state)
            # On-policy: select next action with the current exploration rate
            next_action = self.get_action(next_state, epsilon)
            next_q = self._q_table[next_state][next_action]

        old_q = self._q_table[state][action]
        td_target = reward + self.gamma * next_q
        self._q_table[state][action] = old_q + self.alpha * (td_target - old_q)
