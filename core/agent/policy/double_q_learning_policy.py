import json
import random
import numpy as np
from core.agent.policy.base_policy import TabularPolicy


class DoubleQLearningPolicy(TabularPolicy):
    """
    Off-policy TD control with bias reduction (Double Q-Learning).

    Maintains two independent Q-tables (Q_A and Q_B). At each update step,
    one table is randomly chosen to SELECT the greedy action, and the OTHER
    table is used to EVALUATE that action. This decouples action selection
    from evaluation, eliminating the maximisation bias in vanilla Q-learning.

    Update rule (when table A is selected):
        a* = argmax_a Q_A(s', a)
        Q_A(s, a) += α · [r + γ · Q_B(s', a*) − Q_A(s, a)]
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9):
        super().__init__(alpha, gamma)
        self._q_table_b: dict = {}

    # ------------------------------------------------------------------
    # Override state initialisation to cover both tables
    # ------------------------------------------------------------------

    def _ensure_state(self, state: tuple) -> None:
        if state not in self._q_table:
            self._q_table[state] = [0.0, 0.0, 0.0]
        if state not in self._q_table_b:
            self._q_table_b[state] = [0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # Override value / action queries to use combined tables
    # ------------------------------------------------------------------

    def get_best_action(self, state: tuple) -> int:
        self._ensure_state(state)
        combined = [a + b for a, b in zip(self._q_table[state], self._q_table_b[state])]
        max_val = max(combined)
        return random.choice([i for i, v in enumerate(combined) if v == max_val])

    def get_value(self, state: tuple) -> float:
        self._ensure_state(state)
        combined = [a + b for a, b in zip(self._q_table[state], self._q_table_b[state])]
        return float(max(combined))

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def update(self, state, action: int, reward: float,
               next_state, epsilon: float) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        if random.random() < 0.5:
            # Update Q_A; evaluate with Q_B
            a_star = int(np.argmax(self._q_table[next_state]))
            old_q = self._q_table[state][action]
            eval_q = self._q_table_b[next_state][a_star]
            td_target = reward + self.gamma * eval_q
            self._q_table[state][action] = old_q + self.alpha * (td_target - old_q)
        else:
            # Update Q_B; evaluate with Q_A
            a_star = int(np.argmax(self._q_table_b[next_state]))
            old_q = self._q_table_b[state][action]
            eval_q = self._q_table[next_state][a_star]
            td_target = reward + self.gamma * eval_q
            self._q_table_b[state][action] = old_q + self.alpha * (td_target - old_q)

    # ------------------------------------------------------------------
    # Persistence — save/load both tables
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        serialized = {}
        for state, q_vals in self._q_table.items():
            key = ",".join(map(str, state))
            serialized[key] = {
                "A": q_vals,
                "B": self._q_table_b.get(state, [0.0, 0.0, 0.0])
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4)
        print(f"Saved Double Q-Learning policy to {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._q_table = {}
        self._q_table_b = {}
        for state_str, tables in data.items():
            state = tuple(map(int, state_str.split(",")))
            self._q_table[state] = tables["A"]
            self._q_table_b[state] = tables["B"]
        print(f"Loaded Double Q-Learning policy from {filepath} ({len(self._q_table)} states)")
