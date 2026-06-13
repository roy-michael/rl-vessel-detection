import numpy as np
from core.agent import VesselState

class VesselTrackingRLEnv:
    """
    Reinforcement Learning Environment for Acoustic Vessel Tracking.
    Wraps the frame-by-frame peak/detection association process as an MDP.
    """
    def __init__(self, tracker):
        self.tracker = tracker

    def discretize_state(self, distance_hz, amplitude, score):
        """
        Discretizes continuous detection metrics into a discrete state space tuple.
        Returns state: (dist_bin, amp_bin, tonal_bin)
        """
        # Distance bin
        if distance_hz <= 15.0:
            dist_bin = 0  # Very close
        elif distance_hz <= 45.0:
            dist_bin = 1  # Moderately close
        elif distance_hz <= 90.0:
            dist_bin = 2  # Far but potentially related
        else:
            dist_bin = 3  # Out of range / completely unrelated

        # Amplitude bin
        if amplitude < 0.005:
            amp_bin = 0   # Low amplitude
        elif amplitude < 0.02:
            amp_bin = 1   # Medium amplitude
        else:
            amp_bin = 2   # High amplitude

        # Tonality / Score bin
        if score < 0.45:
            tonal_bin = 0  # Likely noise (below threshold)
        elif score < 0.65:
            tonal_bin = 1  # Moderate tonality
        else:
            tonal_bin = 2  # High tonality

        return (dist_bin, amp_bin, tonal_bin)

    def get_state(self, detection):
        """
        Calculates the current state representation for a given detection.
        """
        centroid = detection['centroid']
        amplitude = detection['amplitude']
        score = detection['score']

        if centroid <= 0:
            # Empty detection state
            return (3, 0, 0)

        # Find closest active vessel track
        active_list = list(self.tracker.active_states.values())
        if not active_list:
            # No active tracks exist
            return (3, self.discretize_state(999.0, amplitude, score)[1], self.discretize_state(999.0, amplitude, score)[2])

        min_dist = float('inf')
        for state in active_list:
            dist = abs(centroid - state.mean_frequency)
            if dist < min_dist:
                min_dist = dist

        return self.discretize_state(min_dist, amplitude, score)

    def get_continuous_state(self, detection):
        """
        Returns a raw continuous state tuple (min_dist_hz, amplitude, score)
        for use with the LinearFAAgent. Does NOT discretise.
        Existing get_state() is untouched and still used by all tabular agents.
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
        Executes the chosen action for a detection at the given time.
        Returns:
            reward (float): The step reward.
            info (dict): Diagnostic info.
        """
        centroid = detection['centroid']
        amplitude = detection['amplitude']
        score = detection['score']

        if centroid <= 0:
            # Invalid/empty detection, force reject
            return 0.0, {"status": "empty_detection"}

        active_list = list(self.tracker.active_states.values())

        # Find closest active track
        closest_state = None
        min_dist = float('inf')
        for state in active_list:
            dist = abs(centroid - state.mean_frequency)
            if dist < min_dist:
                min_dist = dist
                closest_state = state

        # Action 0: REJECT (Ignore as noise)
        if action == 0:
            if min_dist <= 35.0 and score >= 0.45:
                # Penalty for ignoring a strong/close vessel signature
                reward = -10.0
                status = "false_negative"
            else:
                # Correctly ignored noise
                reward = 2.0
                status = "correct_reject"

        # Action 1: ASSOCIATE (Match to closest active vessel)
        elif action == 1:
            if closest_state is None:
                # Cannot associate if no active vessel tracks exist; treat as false association
                reward = -20.0
                status = "invalid_association_no_vessels"
                # Fallback to spawning a new vessel to keep the tracking alive
                self._spawn_vessel(detection, current_time)
            else:
                # Check distance relative to threshold
                if min_dist <= self.tracker.association_threshold_hz:
                    # Good standard association
                    closest_state.add_observation(centroid, detection['spread'], amplitude)
                    self.tracker.last_seen_time[closest_state] = current_time
                    reward = 10.0
                    status = "good_association"
                elif min_dist <= self.tracker.proximity_threshold_hz:
                    # Speed change association: Close old state, start new speed stage under same ID
                    vessel_id = closest_state.vessel_id
                    self.tracker.close_state(closest_state, current_time)
                    
                    new_state = VesselState(current_time, centroid, detection['spread'], initial_amp=amplitude, vessel_id=vessel_id)
                    self.tracker.active_states[new_state] = new_state
                    self.tracker.last_seen_time[new_state] = current_time
                    
                    reward = 5.0
                    status = "speed_change"
                else:
                    # Bad association (too far from active track, causing frequency drift mismatch)
                    # We still associate it (as the action dictates) but give a heavy penalty
                    closest_state.add_observation(centroid, detection['spread'], amplitude)
                    self.tracker.last_seen_time[closest_state] = current_time
                    reward = -15.0
                    status = "bad_association_mismatch"

        # Action 2: SPAWN (Start a new vessel track)
        elif action == 2:
            if closest_state is not None and min_dist <= 35.0:
                # Duplicate track penalty (should have associated with the existing close track)
                reward = -10.0
                status = "duplicate_spawn_penalty"
            else:
                # Correctly spawned new vessel
                if score >= 0.65:
                    reward = 10.0
                else:
                    reward = 5.0
                status = "correct_spawn"

            self._spawn_vessel(detection, current_time)

        else:
            reward = 0.0
            status = "unknown_action"

        return reward, {"status": status, "min_dist": min_dist if min_dist != float('inf') else -1.0}

    def _spawn_vessel(self, detection, current_time):
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
