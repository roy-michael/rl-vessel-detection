import json
import numpy as np
from core.agent.policy.base_policy import Policy

class ActorCriticPolicy(Policy):
    """
    Actor-Critic policy.

    Maintains a Critic (state-value function V(s)) and an Actor (policy preferences theta(s, a)).
    Uses a Softmax function over action preferences to determine action probabilities.
    """

    def __init__(self, alpha_actor: float = 0.05, alpha_critic: float = 0.1, gamma: float = 0.9):
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        
        self.v_table: dict = {}       # state_tuple -> float
        self.theta_table: dict = {}   # state_tuple -> [pref_A0, pref_A1, pref_A2]

    def _ensure_state(self, state: tuple) -> None:
        if state not in self.v_table:
            self.v_table[state] = 0.0
            self.theta_table[state] = [0.0, 0.0, 0.0]

    def get_action(self, state: tuple, epsilon: float) -> int:
        """Selects action based on softmax probabilities, ignores epsilon for pure AC."""
        self._ensure_state(state)
        
        preferences = np.array(self.theta_table[state])
        # Subtract max for numerical stability before exp
        exp_prefs = np.exp(preferences - np.max(preferences))
        probabilities = exp_prefs / np.sum(exp_prefs)
        
        return np.random.choice([0, 1, 2], p=probabilities)

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, epsilon: float) -> None:
        self._ensure_state(state)
        
        # Scale the single-step reward to a small range ([-1.0, 0.5])
        scaled_reward = reward / 20.0
        
        # 1. Evaluate Critic (TD Error)
        v_s = self.v_table[state]
        
        if next_state is None:
            v_next = 0.0
        else:
            self._ensure_state(next_state)
            v_next = self.v_table[next_state]
            
        delta = scaled_reward + self.gamma * v_next - v_s
        
        # Clip TD error to prevent exploding preference values (gradient clipping)
        delta = float(np.clip(delta, -1.0, 1.0))
        
        # 2. Update Critic
        self.v_table[state] += self.alpha_critic * delta

        
        # 3. Update Actor (Policy Gradient)
        preferences = np.array(self.theta_table[state])
        exp_prefs = np.exp(preferences - np.max(preferences))
        pi = exp_prefs / np.sum(exp_prefs)
        
        for a in range(3):
            if a == action:
                self.theta_table[state][a] += self.alpha_actor * delta * (1 - pi[a])
            else:
                self.theta_table[state][a] -= self.alpha_actor * delta * pi[a]

        # Center preferences (normalization) to prevent long-term numerical drift
        centered_prefs = np.array(self.theta_table[state]) - np.mean(self.theta_table[state])
        self.theta_table[state] = list(centered_prefs)

    def get_state(self, det: dict, rl_env) -> tuple:
        """Delegates to the environment's state encoder and returns the discrete representation."""
        state_obj = rl_env.get_state(det)
        if hasattr(state_obj, 'to_discrete'):
            return state_obj.to_discrete()
        return state_obj

    def save(self, filepath: str) -> None:
        serialized = {
            "v_table": {",".join(map(str, k)): float(v) for k, v in self.v_table.items()},
            "theta_table": {",".join(map(str, k)): list(v) for k, v in self.theta_table.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4)
        print(f"Saved policy to {filepath}")

    def load(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Support loading legacy Q-tables into Actor-Critic format gracefully if desired, 
            # or just load AC structure. Assuming AC format:
            if "v_table" in data and "theta_table" in data:
                self.v_table = {
                    tuple(map(int, k.split(","))): float(v)
                    for k, v in data["v_table"].items()
                }
                self.theta_table = {
                    tuple(map(int, k.split(","))): list(v)
                    for k, v in data["theta_table"].items()
                }
                print(f"Loaded Actor-Critic policy from {filepath} ({len(self.v_table)} states)")
            else:
                print(f"Warning: {filepath} does not contain valid Actor-Critic structures. Starting fresh.")
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")

    # Optional Overrides for Diagnostics
    def get_value(self, state: tuple) -> float:
        self._ensure_state(state)
        return float(self.v_table[state])

    def get_best_action(self, state: tuple) -> int:
        self._ensure_state(state)
        preferences = self.theta_table[state]
        max_val = max(preferences)
        # Random choice among ties
        return int(np.random.choice([i for i, v in enumerate(preferences) if v == max_val]))

    @property
    def q_table(self) -> dict:
        """
        Actor-Critic doesn't have a Q-table. For diagnostics, we can return 
        the action preferences as pseudo-Q-values to integrate with existing logs.
        """
        return self.theta_table
