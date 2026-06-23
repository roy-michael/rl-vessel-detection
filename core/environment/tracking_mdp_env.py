import numpy as np
from core.vessel_state import VesselState
from core.environment.mdp_space import TrackingAction, TrackingState, TrackingRewardCalculator

class TrackingMDPEnv:
    """
    Reinforcement Learning Environment for Acoustic Vessel Tracking.

    This class wraps the frame-by-frame acoustic peak detection and state association
    process as a Markov Decision Process (MDP). It coordinates state transitions, 
    manages tracker updates, and delegates reward evaluation to the modular 
    TrackingRewardCalculator.

    Attributes:
        tracker (DSPOrchestrator): The DSP Orchestrator representing the central signal tracking coordinator.
        reward_calculator (TrackingRewardCalculator): Modular component used to evaluate rewards.
    """

    def __init__(self, tracker, reward_calculator=None):
        """
        Initializes the reinforcement learning environment.

        Args:
            tracker (DSPOrchestrator): The central DSP coordinator tracking active vessel stages.
            reward_calculator (TrackingRewardCalculator, optional): The reward evaluator. Defaults to None.
        """
        self.tracker = tracker
        self.reward_calculator = reward_calculator or TrackingRewardCalculator()

    def discretize_state(self, distance_hz, amplitude, score, track_age):
        """
        Discretizes continuous detection metrics into a discrete state space tuple.

        Args:
            distance_hz (float): Distance in Hz to the closest active vessel track.
            amplitude (float): Peak amplitude signature of the acoustic detection.
            score (float): Tonality or confidence score of the peak.
            track_age (float): Continuous track age of the closest active vessel track in seconds.

        Returns:
            Tuple[int, int, int, int]: Discrete state representation (dist_bin, amp_bin, tonal_bin, age_bin).
        """
        # Distance bin (6 bins)
        if distance_hz <= 10.0:
            dist_bin = 0  # Very close
        elif distance_hz <= 25.0:
            dist_bin = 1  # Close
        elif distance_hz <= 45.0:
            dist_bin = 2  # Moderately close
        elif distance_hz <= 70.0:
            dist_bin = 3  # Medium distance
        elif distance_hz <= 100.0:
            dist_bin = 4  # Far but potentially related
        else:
            dist_bin = 5  # Out of range / completely unrelated

        # Amplitude bin (5 bins)
        if amplitude < 0.003:
            amp_bin = 0   # Very low amplitude
        elif amplitude < 0.008:
            amp_bin = 1   # Low amplitude
        elif amplitude < 0.015:
            amp_bin = 2   # Medium amplitude
        elif amplitude < 0.03:
            amp_bin = 3   # High amplitude
        else:
            amp_bin = 4   # Very high amplitude

        # Tonality / Score bin (5 bins)
        if score < 0.35:
            tonal_bin = 0  # Likely noise (below threshold)
        elif score < 0.50:
            tonal_bin = 1  # Low tonality
        elif score < 0.65:
            tonal_bin = 2  # Moderate tonality
        elif score < 0.80:
            tonal_bin = 3  # High tonality
        else:
            tonal_bin = 4  # Very high tonality

        # Track Age bin (4 bins)
        if track_age < 2.0:
            age_bin = 0   # Very young track / potentially temporary noise
        elif track_age < 10.0:
            age_bin = 1   # Young track
        elif track_age < 30.0:
            age_bin = 2   # Established track
        else:
            age_bin = 3   # Long-lived / mature track

        return (dist_bin, amp_bin, tonal_bin, age_bin)

    def get_state(self, detection):
        """
        Calculates and returns the current state representation for a given detection.

        This method extracts features from the detection, computes the distance to the 
        nearest active tracking signal processor, and encapsulates both raw continuous 
        features and discretized state representation inside a TrackingState object.

        Args:
            detection (dict): Dictionary containing the current peak details ('centroid', 'amplitude', 'score').

        Returns:
            TrackingState: The modular state object containing discrete and continuous representations.
        """
        centroid = detection['centroid']
        amplitude = detection['amplitude']
        score = detection['score']

        if centroid <= 0:
            # Empty detection state
            raw_features = (999.0, float(amplitude), float(score), 0.0)
            discrete_tuple = (5, 0, 0, 0)
        else:
            # Find closest active vessel track
            active_list = list(self.tracker.active_states.values())
            if not active_list:
                raw_features = (999.0, float(amplitude), float(score), 0.0)
                discrete_tuple = (5, self.discretize_state(999.0, amplitude, score, 0.0)[1], self.discretize_state(999.0, amplitude, score, 0.0)[2], 0)
            else:
                closest_state = None
                min_dist = float('inf')
                for state in active_list:
                    dist = abs(centroid - state.mean_frequency)
                    if dist < min_dist:
                        min_dist = dist
                        closest_state = state

                track_age = self.tracker.current_time - closest_state.start_time if closest_state else 0.0
                raw_features = (float(min_dist), float(amplitude), float(score), float(track_age))
                discrete_tuple = self.discretize_state(min_dist, amplitude, score, track_age)

        return TrackingState(raw_features, discrete_tuple)

    def get_continuous_state(self, detection):
        """
        Returns a raw continuous state tuple (min_dist_hz, amplitude, score) for compatibility.

        Deprecated: Use get_state() and extract the continuous representation via to_continuous().

        Args:
            detection (dict): Dictionary containing the current peak details.

        Returns:
            Tuple[float, float, float]: Raw continuous features (min_dist, amplitude, score).
        """
        centroid = detection['centroid']
        amplitude = float(detection.get('amplitude', 0.0))
        score = float(detection.get('score', 0.0))

        if centroid <= 0:
            return (999.0, amplitude, score)

        active_list = list(self.tracker.active_states.values())
        if not active_list:
            return (999.0, amplitude, score)

        min_dist = min(abs(centroid - s.mean_frequency) for s in active_list)
        return (float(min_dist), amplitude, score)

    def step(self, action, detection, current_time):
        """
        Executes the transition and tracking updates corresponding to the chosen action.

        Args:
            action (int or TrackingAction): The decision action (0=REJECT, 1=ASSOCIATE, 2=SPAWN).
            detection (dict): Dictionary containing current acoustic peak metrics.
            current_time (float): Current absolute timeline timestamp.

        Returns:
            reward (float): Decoupled step reward evaluated by the TrackingRewardCalculator.
            info (dict): Diagnostic dictionary detailing transition status and proximity.
        """
        centroid = detection['centroid']
        amplitude = detection['amplitude']
        score = detection['score']

        if centroid <= 0:
            # Invalid/empty detection, force reject
            return 0.0, {"status": "empty_detection"}

        # Map integer action to TrackingAction enum if necessary
        action = TrackingAction(action)
        active_list = list(self.tracker.active_states.values())

        # Find closest active track
        closest_state = None
        min_dist = float('inf')
        for state in active_list:
            dist = abs(centroid - state.mean_frequency)
            if dist < min_dist:
                min_dist = dist
                closest_state = state

        closest_state_exists = (closest_state is not None)

        # Delegate reward calculation to modular TrackingRewardCalculator
        reward, status = self.reward_calculator.calculate_reward(
            action=action,
            min_dist=min_dist,
            score=score,
            closest_state_exists=closest_state_exists,
            association_threshold_hz=self.tracker.association_threshold_hz,
            proximity_threshold_hz=self.tracker.proximity_threshold_hz
        )

        # Execute target tracking updates and state changes
        if action == TrackingAction.ASSOCIATE:
            if closest_state is None:
                # Fallback to spawning a new vessel to keep the tracking alive
                self._spawn_vessel(detection, current_time)
            else:
                if min_dist <= self.tracker.association_threshold_hz:
                    # Good standard association
                    closest_state.add_observation(centroid, detection['spread'], amplitude)
                    self.tracker.last_seen_time[closest_state] = current_time
                elif min_dist <= self.tracker.proximity_threshold_hz:
                    # Speed change: Close old state, start new stage under same ID
                    vessel_id = closest_state.vessel_id
                    self.tracker.close_state(closest_state, current_time)
                    new_state = VesselState(current_time, centroid, detection['spread'], initial_amp=amplitude, vessel_id=vessel_id)
                    self.tracker.active_states[new_state] = new_state
                    self.tracker.last_seen_time[new_state] = current_time
                else:
                    # Bad association (mismatch frequency drift)
                    closest_state.add_observation(centroid, detection['spread'], amplitude)
                    self.tracker.last_seen_time[closest_state] = current_time

        elif action == TrackingAction.SPAWN:
            self._spawn_vessel(detection, current_time)

        return reward, {"status": status, "min_dist": min_dist if min_dist != float('inf') else -1.0}

    def _spawn_vessel(self, detection, current_time):
        """
        Spawns a new active vessel state track.

        Args:
            detection (dict): Peak detection dictionary.
            current_time (float): Current absolute timestamp.
        """
        self.tracker.vessel_counter += 1
        vid = f"Vessel {self.tracker.vessel_counter}"
        new_state = VesselState(
            current_time, 
            detection['centroid'], 
            detection['spread'], 
            initial_amp=detection['amplitude'], 
            vessel_id=vid
        )
        self.tracker.active_states[new_state] = new_state
        self.tracker.last_seen_time[new_state] = current_time
