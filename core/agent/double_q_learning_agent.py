import json
import random
import numpy as np
from core.agent.base_agent import TDBaseAgent

class DoubleQLearningAgent(TDBaseAgent):
    """
    Double Q-Learning Agent (Lesson 5 extension — off-policy with bias reduction).

    Maintains two independent Q-tables (Q_A and Q_B). At each update step, one
    table is randomly chosen to SELECT the greedy action, and the OTHER table is
    used to EVALUATE that action. This decoupling eliminates the maximisation bias
    inherent in vanilla Q-learning, where argmax and evaluation use the same table.

    Update rule (when table A is selected for update):
        a* = argmax_{a} Q_A(s', a)
        Q_A(s, a) += alpha * [r + gamma * Q_B(s', a*) - Q_A(s, a)]
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.15):
        super().__init__(alpha, gamma, epsilon)
        self.q_table_b = {}  # Second independent Q-table

    def _ensure_state(self, state):
        """Initialises both tables for unseen states."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        if state not in self.q_table_b:
            self.q_table_b[state] = [0.0, 0.0, 0.0]

    def get_best_action(self, state):
        """Greedy action using the *sum* of both tables (evaluation policy)."""
        self._ensure_state(state)
        combined = [a + b for a, b in zip(self.q_table[state], self.q_table_b[state])]
        max_val = max(combined)
        best = [i for i, v in enumerate(combined) if v == max_val]
        return random.choice(best)

    def get_value(self, state):
        self._ensure_state(state)
        combined = [a + b for a, b in zip(self.q_table[state], self.q_table_b[state])]
        return float(max(combined))

    def learn(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        # Randomly assign update to table A or B
        if random.random() < 0.5:
            # Update Q_A; evaluate with Q_B
            a_star = int(np.argmax(self.q_table[state]))
            old_q = self.q_table[state][action]
            eval_q = self.q_table_b[next_state][a_star]
            td_target = reward + self.gamma * eval_q
            self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)
        else:
            # Update Q_B; evaluate with Q_A
            a_star = int(np.argmax(self.q_table_b[state]))
            old_q = self.q_table_b[state][action]
            eval_q = self.q_table[next_state][a_star]
            td_target = reward + self.gamma * eval_q
            self.q_table_b[state][action] = old_q + self.alpha * (td_target - old_q)

    def save_policy(self, filepath):
        """Saves both Q-tables to a JSON file."""
        serialized = {}
        for state, q_vals in self.q_table.items():
            key = ",".join(map(str, state))
            serialized[key] = {"A": q_vals, "B": self.q_table_b.get(state, [0.0, 0.0, 0.0])}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4)
        print(f"Saved Double Q-Learning policy to {filepath}")

    def load_policy(self, filepath):
        """Loads both Q-tables from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            serialized = json.load(f)
        self.q_table = {}
        self.q_table_b = {}
        for state_str, tables in serialized.items():
            state = tuple(map(int, state_str.split(",")))
            self.q_table[state] = tables["A"]
            self.q_table_b[state] = tables["B"]
        print(f"Loaded Double Q-Learning policy from {filepath} (States loaded: {len(self.q_table)})")
