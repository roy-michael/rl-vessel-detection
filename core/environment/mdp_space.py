from enum import IntEnum
from typing import Any, Dict, Tuple


class TrackingAction(IntEnum):
    """
    Representation of the high-level decision actions available to the tracking agent.

    Attributes:
        REJECT: Ignore the detection peak as ambient noise or clutter.
        ASSOCIATE: Assign the peak to the nearest active tracking signal processor.
        SPAWN: Spawn a new tracking signal processor representing a newly discovered target.
    """
    REJECT = 0
    ASSOCIATE = 1
    SPAWN = 2


class TrackingState:
    """
    State wrapper enclosing the feature representation of the tracking environment.

    Purpose:
        Standardizes discrete bin-states (tabular agents) and continuous features (function approximation).
    """

    def __init__(self, raw_features: Tuple[float, float, float, float], discrete_tuple: Tuple[int, int, int, int] = None):
        """
        Args:
            raw_features: A tuple containing (distance_hz, amplitude, tonality_score, track_age)
            discrete_tuple: The discretized representation of features for tabular Q-learning
        """
        self.distance_hz, self.amplitude, self.tonality, self.track_age = raw_features
        self.discrete = discrete_tuple

    def to_discrete(self) -> Tuple[int, int, int, int]:
        """Returns the discrete state tuple representation for Q-tables."""
        return self.discrete

    def to_continuous(self) -> Tuple[float, float, float, float]:
        """Returns the raw continuous features for function approximation."""
        return (self.distance_hz, self.amplitude, self.tonality, self.track_age)

    def __repr__(self) -> str:
        return f"TrackingState(Discrete: {self.discrete}, Continuous: ({self.distance_hz:.1f}, {self.amplitude:.4f}, {self.tonality:.2f}, {self.track_age:.1f}s))"


class TrackingRewardCalculator:
    """
    Class responsible for calculating learning step rewards for the vessel tracking task.

    Purpose:
        Decouples the reward structure, scales, and criteria into a modular object that can be
        customized independently of environment progression or signal parsing.

    Rationale for Reward Selection:
        The numerical values were selected to create a balanced, zero-sum-like economy 
        that heavily penalizes illegal state transitions and strongly rewards continuity:
        
        - `+10.0` (Good Association / High Tonal Spawn): The highest possible reward. The primary goal 
          of the tracker is to maintain continuous tracks and identify clear targets.
        - `-10.0` (False Negative / Duplicate Spawn): A symmetric penalty. Ignoring a valid track or 
          cluttering the DSP layer with duplicates is equally as bad as succeeding is good.
        - `-20.0` (Invalid Association): The most severe penalty. Attempting to associate to a 
          non-existent track is an illegal state transition. It is penalized doubly to rapidly 
          discourage this path during early exploration.
        - `+5.0` (Speed Change / Med Tonal Spawn): A moderate reward. We want to encourage spawning, 
          but it must be strictly lower than +10.0 so the agent prefers tracking an existing vessel 
          over constantly spawning fragmented new tracks.
        - `+2.0` (Correct Reject): A small positive baseline reward for filtering noise. If this was 
          too high, the agent would learn a lazy policy of rejecting everything.
    """

    def __init__(self, reject_penalty: float = -10.0, correct_reject_reward: float = 2.0,
                 good_assoc_reward: float = 10.0, speed_change_reward: float = 5.0,
                 mismatch_penalty: float = -15.0, invalid_assoc_penalty: float = -20.0,
                 dup_spawn_penalty: float = -10.0, spawn_vessel_high_tonal: float = 10.0,
                 spawn_vessel_med_tonal: float = 5.0):
        self.reject_penalty = reject_penalty
        self.correct_reject_reward = correct_reject_reward
        self.good_assoc_reward = good_assoc_reward
        self.speed_change_reward = speed_change_reward
        self.mismatch_penalty = mismatch_penalty
        self.invalid_assoc_penalty = invalid_assoc_penalty
        self.dup_spawn_penalty = dup_spawn_penalty
        self.spawn_vessel_high_tonal = spawn_vessel_high_tonal
        self.spawn_vessel_med_tonal = spawn_vessel_med_tonal

    def calculate_reward(self, action: TrackingAction, min_dist: float, score: float, closest_state_exists: bool, association_threshold_hz: float, proximity_threshold_hz: float) -> Tuple[float, str]:
        """
        Calculates step reward based on the selected action, state variables, and matching metrics.

        Returns:
            Tuple[reward_value, outcome_status_string]
        """
        # Action 0: REJECT
        if action == TrackingAction.REJECT:
            if min_dist <= 35.0 and score >= 0.45:
                return self.reject_penalty, "false_negative"
            else:
                return self.correct_reject_reward, "correct_reject"

        # Action 1: ASSOCIATE
        elif action == TrackingAction.ASSOCIATE:
            if not closest_state_exists:
                return self.invalid_assoc_penalty, "invalid_association_no_vessels"
            else:
                if min_dist <= association_threshold_hz:
                    return self.good_assoc_reward, "good_association"
                elif min_dist <= proximity_threshold_hz:
                    return self.speed_change_reward, "speed_change"
                else:
                    return self.mismatch_penalty, "bad_association_mismatch"

        # Action 2: SPAWN
        elif action == TrackingAction.SPAWN:
            if closest_state_exists and min_dist <= 35.0:
                return self.dup_spawn_penalty, "duplicate_spawn_penalty"
            else:
                if score >= 0.65:
                    return self.spawn_vessel_high_tonal, "correct_spawn"
                else:
                    return self.spawn_vessel_med_tonal, "correct_spawn"

        return 0.0, "unknown_action"
