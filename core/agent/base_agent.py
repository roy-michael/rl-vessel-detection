import json
import random
import numpy as np

class TDBaseAgent:
    """
    Base class for Temporal Difference learning agents.
    Handles the state representation, Q-table, value functions, policy, and persistence.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.15):
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        self.q_table = {}        # state_tuple -> [Q(s, a0), Q(s, a1), Q(s, a2)]

    def _ensure_state(self, state):
        """Initializes Q-values for a state to zeros if not already visited."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        self._ensure_state(state)
        return self.q_table[state][action]

    def get_value(self, state):
        """
        Calculates the State-Value Function V(s) = max_a Q(s, a).
        """
        self._ensure_state(state)
        return float(np.max(self.q_table[state]))

    def get_best_action(self, state):
        """
        Returns the greedy action for a state: argmax_a Q(s, a).
        """
        self._ensure_state(state)
        q_values = self.q_table[state]
        # To break ties randomly
        max_val = np.max(q_values)
        actions_with_max = [i for i, val in enumerate(q_values) if val == max_val]
        return random.choice(actions_with_max)

    def get_action(self, state, epsilon=None):
        """
        Policy: Epsilon-Greedy Action Selection.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            # Exploration
            return random.randint(0, 2)
        else:
            # Exploitation
            return self.get_best_action(state)

    def save_policy(self, filepath):
        """
        Saves the Q-table to a JSON file.
        Converts the state tuple keys to string format: "dist,amp,tonal"
        """
        serialized_q_table = {}
        for state, q_values in self.q_table.items():
            state_key = ",".join(map(str, state))
            serialized_q_table[state_key] = q_values

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized_q_table, f, indent=4)
        print(f"Saved policy to {filepath}")

    def load_policy(self, filepath):
        """
        Loads the Q-table from a JSON file.
        Reconstructs the state tuple keys.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            serialized_q_table = json.load(f)

        self.q_table = {}
        for state_str, q_values in serialized_q_table.items():
            state = tuple(map(int, state_str.split(",")))
            self.q_table[state] = q_values
        print(f"Loaded policy from {filepath} (States loaded: {len(self.q_table)})")
