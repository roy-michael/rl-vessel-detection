import random
import numpy as np
from core.agent.policy.base_policy import Policy


class LinearFAPolicy(Policy):
    """
    Linear Function Approximation with Tile Coding.

    Approximates Q(s, a) = w[a] · φ(s), where φ(s) is a sparse binary
    feature vector produced by overlapping tile codings over the continuous
    state (min_dist_hz, amplitude, score).

    This policy generalises to unseen states without explicit discretisation.
    TD update: w[a] += α · δ · φ(s)

    Unlike tabular policies, get_state() requests the *continuous* state from
    the RL environment, so callers need not know which state encoding to use.
    """

    N_ACTIONS = 3

    def __init__(self, alpha: float = 0.01, gamma: float = 0.9,
                 n_tilings: int = 4, n_tiles: int = 6,
                 dist_range: tuple = (0.0, 200.0),
                 amp_range: tuple = (0.0, 0.1),
                 score_range: tuple = (0.0, 1.0),
                 age_range: tuple = (0.0, 60.0)):
        self.alpha = alpha / n_tilings   # Normalise by tiling count
        self.gamma = gamma
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.dist_range = dist_range
        self.amp_range = amp_range
        self.score_range = score_range
        self.age_range = age_range

        self.n_features = n_tilings * (n_tiles ** 4)
        self.weights = np.zeros((self.N_ACTIONS, self.n_features))

    # ------------------------------------------------------------------
    # Tile coding helpers
    # ------------------------------------------------------------------

    def _tile_indices(self, state: tuple) -> list:
        """Active tile indices for a continuous state (dist, amp, score, age)."""
        dist, amp, score, age = state
        indices = []
        tiling_size = self.n_tiles ** 4

        for tiling in range(self.n_tilings):
            offset = tiling / self.n_tilings

            def tile_idx(val, lo, hi):
                normalised = (val - lo) / (hi - lo + 1e-9)
                return int(min(self.n_tiles - 1,
                               max(0, (normalised + offset) * self.n_tiles % self.n_tiles)))

            d_idx = tile_idx(dist, *self.dist_range)
            a_idx = tile_idx(amp,  *self.amp_range)
            s_idx = tile_idx(score, *self.score_range)
            g_idx = tile_idx(age, *self.age_range)
            flat_idx = (tiling * tiling_size
                        + d_idx * (self.n_tiles ** 3)
                        + a_idx * (self.n_tiles ** 2)
                        + s_idx * self.n_tiles
                        + g_idx)
            indices.append(flat_idx)
        return indices

    def _phi(self, state: tuple) -> np.ndarray:
        """Sparse binary feature vector φ(s)."""
        phi = np.zeros(self.n_features)
        for idx in self._tile_indices(state):
            phi[idx] = 1.0
        return phi

    def _q_value(self, state: tuple, action: int) -> float:
        return float(np.dot(self.weights[action], self._phi(state)))

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def get_best_action(self, state: tuple) -> int:
        q_vals = [self._q_value(state, a) for a in range(self.N_ACTIONS)]
        max_q = max(q_vals)
        return random.choice([i for i, v in enumerate(q_vals) if v == max_q])

    def get_value(self, state: tuple) -> float:
        return max(self._q_value(state, a) for a in range(self.N_ACTIONS))

    def get_action(self, state: tuple, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        return self.get_best_action(state)

    def get_state(self, det: dict, rl_env) -> tuple:
        """Returns the *continuous* state representation from the TrackingState."""
        state_obj = rl_env.get_state(det)
        if hasattr(state_obj, 'to_continuous'):
            return state_obj.to_continuous()
        return state_obj

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, epsilon: float) -> None:
        """Semi-gradient TD(0) weight update."""
        q_sa = self._q_value(state, action)
        if next_state is None:
            max_q_next = 0.0
        else:
            max_q_next = max(self._q_value(next_state, a) for a in range(self.N_ACTIONS))
        td_error = reward + self.gamma * max_q_next - q_sa
        phi = self._phi(state)
        self.weights[action] += self.alpha * td_error * phi

    # ------------------------------------------------------------------
    # Persistence — uses .npy (not JSON)
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        np.save(filepath, self.weights)
        print(f"Saved LinearFA policy weights to {filepath}")

    def load(self, filepath: str) -> None:
        loaded_weights = np.load(filepath)
        if loaded_weights.shape != self.weights.shape:
            raise ValueError(f"Shape mismatch: loaded {loaded_weights.shape}, expected {self.weights.shape}")
        self.weights = loaded_weights
        print(f"Loaded LinearFA policy weights from {filepath} (shape: {self.weights.shape})")
