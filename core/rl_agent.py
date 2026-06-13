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


class QLearningAgent(TDBaseAgent):
    """
    Q-Learning Agent (Off-policy TD control).
    Updates Q-values using the maximum possible utility of the next state:
    Q(s, a) = Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def learn(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Q-learning temporal difference target
        td_target = reward + self.gamma * max_future_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)


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


# =============================================================================
# NEW AGENTS — Course-aligned extensions (existing agents above are untouched)
# =============================================================================

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


class DynaQAgent(TDBaseAgent):
    """
    Dyna-Q Agent (Lesson 7 — Planning & Models).

    Extends Q-Learning by maintaining a deterministic environment model
    Model[(s, a)] -> (next_s, reward). After each real experience, the agent
    performs n_planning simulated planning updates drawn uniformly from the
    previously observed state-action pairs. This greatly improves sample
    efficiency because the agent effectively replays and learns from past
    transitions multiple times per real step.

    Update sequence per real step:
      1. Real interaction:  Q(s, a) <- Q-learning update
      2. Model update:      Model[(s, a)] <- (s', r)
      3. Planning loop (n_planning times):
           Pick (s_i, a_i) uniformly from model keys
           (s', r) <- Model[(s_i, a_i)]
           Q(s_i, a_i) <- Q-learning update
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.15, n_planning=20):
        super().__init__(alpha, gamma, epsilon)
        self.n_planning = n_planning
        self.model = {}          # {(state, action): (next_state, reward)}
        self._model_keys = []    # Ordered list for O(1) random sampling

    def learn(self, state, action, reward, next_state):
        # --- Step 1: Real Q-learning update ---
        self._ensure_state(state)
        self._ensure_state(next_state)
        old_q = self.q_table[state][action]
        max_future_q = float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * max_future_q
        self.q_table[state][action] = old_q + self.alpha * (td_target - old_q)

        # --- Step 2: Model update ---
        key = (state, action)
        if key not in self.model:
            self._model_keys.append(key)
        self.model[key] = (next_state, reward)

        # --- Step 3: Planning loop ---
        if len(self._model_keys) == 0:
            return
        n = min(self.n_planning, len(self._model_keys))
        sample_keys = random.choices(self._model_keys, k=n)
        for sim_state, sim_action in sample_keys:
            sim_next_state, sim_reward = self.model[(sim_state, sim_action)]
            self._ensure_state(sim_state)
            self._ensure_state(sim_next_state)
            sim_old_q = self.q_table[sim_state][sim_action]
            sim_max_q = float(np.max(self.q_table[sim_next_state]))
            sim_target = sim_reward + self.gamma * sim_max_q
            self.q_table[sim_state][sim_action] = sim_old_q + self.alpha * (sim_target - sim_old_q)
