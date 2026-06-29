import json
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """
    Abstract base class for all RL update rules.

    A Policy owns the update rule (learning algorithm), the value function or
    weight parameters, and knows how to encode observations into states.
    The RLAgent owns one Policy and delegates all algorithm-specific logic to it.
    """

    @abstractmethod
    def get_action(self, state: Any, epsilon: float) -> int:
        """ε-greedy action selection."""

    @abstractmethod
    def update(self, state: Any, action: int, reward: float,
               next_state: Any, epsilon: float) -> None:
        """
        Incorporate a (s, a, r, s') transition.
        epsilon is passed so on-policy methods (e.g. SARSA) can select
        next_action internally without needing external coordination.
        """

    @abstractmethod
    def get_state(self, det: dict, rl_env) -> Any:
        """
        Extract the state representation for this policy from a detection dict
        and the RL environment. Tabular policies use discrete bins;
        LinearFAPolicy uses a continuous tuple.
        """

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist learned parameters to disk."""

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Restore learned parameters from disk."""

    # ------------------------------------------------------------------
    # Optional overrides — used for diagnostics and value queries
    # ------------------------------------------------------------------

    def get_value(self, state: Any) -> float:
        """V(s) = max_a Q(s, a). Override for policies that track values."""
        return 0.0

    def get_best_action(self, state: Any) -> int:
        """Greedy action. Override for diagnostics."""
        return 0

    @property
    def q_table(self) -> dict:
        """Q-table access for diagnostics. Returns {} for FA-based policies."""
        return {}


# ---------------------------------------------------------------------------
# Shared base for the four tabular (Q-table) algorithms
# ---------------------------------------------------------------------------

class TabularPolicy(Policy):
    """
    Shared foundation for Q-table policies.
    Provides: Q-table management, ε-greedy action selection, greedy value /
    action queries, discrete state encoding, and JSON save/load.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9):
        self.alpha = alpha
        self.gamma = gamma
        self._q_table: dict = {}  # state_tuple -> [Q(s,A0), Q(s,A1), Q(s,A2)]

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _ensure_state(self, state: tuple) -> None:
        if state not in self._q_table:
            self._q_table[state] = [0.0, 0.0, 0.0]

    def get_best_action(self, state: tuple) -> int:
        self._ensure_state(state)
        q = self._q_table[state]
        max_val = max(q)
        return random.choice([i for i, v in enumerate(q) if v == max_val])

    def get_value(self, state: tuple) -> float:
        self._ensure_state(state)
        return float(np.max(self._q_table[state]))

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_action(self, state: tuple, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 2)
        return self.get_best_action(state)

    def get_state(self, det: dict, rl_env) -> tuple:
        """Delegates to the environment's state encoder and returns the discrete representation."""
        state_obj = rl_env.get_state(det)
        if hasattr(state_obj, 'to_discrete'):
            return state_obj.to_discrete()
        return state_obj

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        serialized = {
            ",".join(map(str, k)): v
            for k, v in self._q_table.items()
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=4)
        print(f"Saved policy to {filepath}")

    def load(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._q_table = {
            tuple(map(int, k.split(","))): v
            for k, v in data.items()
        }
        print(f"Loaded policy from {filepath} ({len(self._q_table)} states)")

    # ------------------------------------------------------------------
    # Diagnostic property
    # ------------------------------------------------------------------

    @property
    def q_table(self) -> dict:
        return self._q_table
