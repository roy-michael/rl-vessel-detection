from typing import Any
from core.agent.policy.base_policy import Policy


class RLAgent:
    """
    A single RL agent with a swappable learning policy.

    The agent owns the exploration rate (epsilon) and delegates all
    algorithm-specific logic — update rules, value functions, state encoding,
    and parameter persistence — to its Policy object.

    Usage
    -----
    >>> from core.agent.policy import QLearningPolicy
    >>> agent = RLAgent(policy=QLearningPolicy(alpha=0.15, gamma=0.85), epsilon=0.4)
    >>> state  = agent.observe(det, rl_env)
    >>> action = agent.act(state)
    >>> reward, info = rl_env.step(action, det, current_time)
    >>> next_state = agent.observe(det, rl_env)
    >>> agent.step(state, action, reward, next_state)   # no-op at epsilon=0
    """

    def __init__(self, policy: Policy, epsilon: float = 0.15):
        self.policy = policy
        self.epsilon = epsilon   # Mutable: set per-episode during training

    # ------------------------------------------------------------------
    # Core interaction loop
    # ------------------------------------------------------------------

    def observe(self, det: dict, rl_env) -> Any:
        """
        Encode a detection dict into a state representation.
        Delegates to the policy so that tabular policies use discrete bins
        and LinearFAPolicy uses continuous tile-coded features.
        """
        return self.policy.get_state(det, rl_env)

    def act(self, state: Any) -> int:
        """ε-greedy action selection using the current exploration rate."""
        return self.policy.get_action(state, self.epsilon)

    def step(self, state: Any, action: int,
             reward: float, next_state: Any) -> None:
        """
        Update the policy from a (s, a, r, s') transition.
        No-op during evaluation (epsilon == 0) to preserve the loaded policy.
        """
        if self.epsilon > 0.0:
            self.policy.update(state, action, reward, next_state, self.epsilon)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_policy(self, filepath: str) -> None:
        self.policy.save(filepath)

    def load_policy(self, filepath: str) -> None:
        self.policy.load(filepath)

    # ------------------------------------------------------------------
    # Diagnostic pass-throughs (used by train_rl.py value summaries)
    # ------------------------------------------------------------------

    def get_value(self, state: Any) -> float:
        return self.policy.get_value(state)

    def get_best_action(self, state: Any) -> int:
        return self.policy.get_best_action(state)

    @property
    def q_table(self) -> dict:
        """Q-table access for diagnostics. Empty dict for FA-based policies."""
        return self.policy.q_table
