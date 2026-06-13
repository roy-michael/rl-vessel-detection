import random
import numpy as np

class LinearFAAgent:
    """
    Linear Function Approximation Agent with Tile Coding (Lesson 6).

    Instead of a lookup table, approximates Q(s, a) = w[a] · phi(s), where
    phi(s) is a binary tile-coded feature vector over the continuous state
    (min_dist_hz, amplitude, score). This generalises to unseen states without
    requiring explicit discretisation.

    Uses n_tilings overlapping grids with n_tiles divisions per dimension.
    TD update: w[a] += alpha * delta * phi(s)

    The 'state' expected by learn/get_action is a raw continuous tuple
    (min_dist_hz, amplitude, score) returned by rl_env.get_continuous_state().
    """
    N_ACTIONS = 3

    def __init__(self, alpha=0.01, gamma=0.9, epsilon=0.15,
                 n_tilings=4, n_tiles=8,
                 dist_range=(0.0, 200.0),
                 amp_range=(0.0, 0.1),
                 score_range=(0.0, 1.0)):
        self.alpha = alpha / n_tilings  # Normalise learning rate by number of tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.dist_range = dist_range
        self.amp_range = amp_range
        self.score_range = score_range

        # Total feature vector length = n_tilings * n_tiles^3
        self.n_features = n_tilings * (n_tiles ** 3)
        # Weight vectors: one per action
        self.weights = np.zeros((self.N_ACTIONS, self.n_features))

    # ------------------------------------------------------------------
    # Tile coding helpers
    # ------------------------------------------------------------------
    def _tile_indices(self, state):
        """Returns active tile indices for continuous state (dist, amp, score)."""
        dist, amp, score = state
        indices = []
        tiling_size = self.n_tiles ** 3

        for tiling in range(self.n_tilings):
            # Offset each tiling by a fraction of tile width to create overlapping coverage
            offset = tiling / self.n_tilings

            def tile_idx(val, lo, hi):
                normalised = (val - lo) / (hi - lo + 1e-9)
                return int(min(self.n_tiles - 1, max(0, (normalised + offset) * self.n_tiles % self.n_tiles)))

            d_idx = tile_idx(dist, *self.dist_range)
            a_idx = tile_idx(amp, *self.amp_range)
            s_idx = tile_idx(score, *self.score_range)

            flat_idx = tiling * tiling_size + d_idx * (self.n_tiles ** 2) + a_idx * self.n_tiles + s_idx
            indices.append(flat_idx)
        return indices

    def _phi(self, state):
        """Returns the sparse binary feature vector phi(s)."""
        phi = np.zeros(self.n_features)
        for idx in self._tile_indices(state):
            phi[idx] = 1.0
        return phi

    def _q_value(self, state, action):
        """Approximated Q-value: w[a] · phi(s)."""
        return float(np.dot(self.weights[action], self._phi(state)))

    # ------------------------------------------------------------------
    # Agent interface (mirrors TDBaseAgent)
    # ------------------------------------------------------------------
    def get_best_action(self, state):
        q_vals = [self._q_value(state, a) for a in range(self.N_ACTIONS)]
        max_q = max(q_vals)
        best = [i for i, v in enumerate(q_vals) if v == max_q]
        return random.choice(best)

    def get_value(self, state):
        return max(self._q_value(state, a) for a in range(self.N_ACTIONS))

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.N_ACTIONS - 1)
        return self.get_best_action(state)

    def learn(self, state, action, reward, next_state):
        """Semi-gradient TD(0) weight update."""
        q_sa = self._q_value(state, action)
        max_q_next = max(self._q_value(next_state, a) for a in range(self.N_ACTIONS))
        td_error = reward + self.gamma * max_q_next - q_sa
        phi = self._phi(state)
        self.weights[action] += self.alpha * td_error * phi

    def save_policy(self, filepath):
        """Saves weight matrix to a .npy file (filepath with .npy extension)."""
        np.save(filepath, self.weights)
        print(f"Saved LinearFA policy weights to {filepath}")

    def load_policy(self, filepath):
        """Loads weight matrix from a .npy file."""
        self.weights = np.load(filepath)
        print(f"Loaded LinearFA policy weights from {filepath} (shape: {self.weights.shape})")
