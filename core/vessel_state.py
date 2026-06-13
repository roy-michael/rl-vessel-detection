import numpy as np
import inspect

class VesselState:
    def __init__(self, start_time, initial_freq, initial_spread, initial_amp=0.0, vessel_id=None):
        self.start_time = start_time
        self.end_time = None
        self.frequencies = [initial_freq]
        self.spreads = [initial_spread]
        self.amplitudes = [initial_amp]
        self.vessel_id = vessel_id

    def add_observation(self, freq, spread, amp):
        self.frequencies.append(freq)
        self.spreads.append(spread)
        self.amplitudes.append(amp)

    def close(self, end_time):
        self.end_time = end_time

    @property
    def mean_frequency(self):
        return float(np.mean(self.frequencies))

    @property
    def total_variance(self):
        drift_variance = float(np.var(self.frequencies))
        inner_variance = float(np.mean(np.array(self.spreads) ** 2))
        return drift_variance + inner_variance


class VesselStateTracker:
    def __init__(self, drift_threshold_hz=35.0, min_vessel_score=0.45, min_duration_sec=10.0, proximity_threshold_hz=65.0, association_threshold_hz=80.0, consolidation_threshold_hz=65.0):
        self.drift_threshold_hz = drift_threshold_hz
        self.min_vessel_score = min_vessel_score
        self.min_duration_sec = min_duration_sec
        self.proximity_threshold_hz = proximity_threshold_hz
        self.association_threshold_hz = association_threshold_hz
        self.consolidation_threshold_hz = consolidation_threshold_hz
        
        self.states = []                  # completed/archived states
        self.active_states = {}           # active_state_obj -> active_state_obj
        self.last_seen_time = {}          # active_state_obj -> float (last seen timestamp)
        self.vessel_counter = 0

        # RL policy attributes
        self.q_agent = None
        self.rl_env = None
        self.rl_epsilon = 0.0
        self.rl_stats = None
        self.rl_history = []

    @property
    def current_state(self):
        if self.active_states:
            return next(iter(self.active_states.values()))
        return None

    def get_vessel_fundamental(self, vessel_id):
        if not vessel_id or vessel_id == "Noise":
            return None
        all_states = self.states + list(self.active_states.values())
        v_states = [s for s in all_states if s.vessel_id == vessel_id]
        if not v_states:
            return None
        return min(s.mean_frequency for s in v_states)

    def cluster_active_states(self, current_time):
        """
        Runs a real-time spectral clustering over active states to merge duplicates
        and harmonics under the same Vessel ID, based on their fundamental frequencies.
        """
        active_list = list(self.active_states.values())
        n = len(active_list)
        if n < 2:
            return
            
        merged_any = False
        
        for i in range(n):
            for j in range(i + 1, n):
                stateA = active_list[i]
                stateB = active_list[j]
                
                # Skip if they already have the same Vessel ID
                if stateA.vessel_id == stateB.vessel_id:
                    continue
                    
                fundA = self.get_vessel_fundamental(stateA.vessel_id)
                fundB = self.get_vessel_fundamental(stateB.vessel_id)
                if fundA is None or fundB is None:
                    continue
                    
                should_merge = False
                reason = ""
                
                # Proximity check of fundamentals (same physical vessel split across NMF components)
                if abs(fundA - fundB) <= self.proximity_threshold_hz:
                    should_merge = True
                    reason = f"fundamental proximity ({fundB:.1f}Hz vs {fundA:.1f}Hz)"
                else:
                    # Harmonic check of fundamentals (integer multiples check)
                    f1 = min(fundA, fundB)
                    f2 = max(fundA, fundB)
                    if 10.0 < f1 <= 150.0 and f2 <= 500.0:
                        ratio = f2 / f1
                        k = round(ratio)
                        if 2 <= k <= 8:
                            expected_f = k * f1
                            if abs(f2 - expected_f) <= self.proximity_threshold_hz:
                                should_merge = True
                                reason = f"fundamental harmonic relationship (k={k}, {f2:.1f}Hz is harmonic of fundamental {f1:.1f}Hz)"
                            
                if should_merge:
                    # Determine which ID to keep (the older one / smaller counter index)
                    idA = stateA.vessel_id
                    idB = stateB.vessel_id
                    
                    try:
                        numA = int(idA.split()[-1])
                        numB = int(idB.split()[-1])
                        keep_id = idA if numA < numB else idB
                        discard_id = idB if numA < numB else idA
                    except Exception:
                        keep_id = idA
                        discard_id = idB
                        
                    # Perform the merge!
                    stateA.vessel_id = keep_id
                    stateB.vessel_id = keep_id
                    
                    # Update all historical completed states
                    merge_count = 0
                    for s in self.states:
                        if s.vessel_id == discard_id:
                            s.vessel_id = keep_id
                            merge_count += 1
                            
                    print(f"\n>>> [{current_time:.1f}s] Merged {discard_id} into {keep_id} due to {reason} (Merged {merge_count} archived stages)")
                    merged_any = True
                    break
            if merged_any:
                break
                
    def consolidate_all_vessels(self, current_time):
        """
        Consolidates different Vessel IDs together across all completed and active states
        if their fundamental frequencies are within consolidation_threshold_hz
        OR if they exhibit a harmonic relationship (using a frequency-drift tolerant threshold).
        Runs iteratively until no more merges are possible.
        """
        while True:
            active_list = list(self.active_states.values())
            all_states = self.states + active_list
            if not all_states:
                break

            # 1. Group states by Vessel ID
            vessel_groups = {}
            for state in all_states:
                vid = state.vessel_id
                if not vid or vid == "Noise":
                    continue
                vessel_groups.setdefault(vid, []).append(state)

            if len(vessel_groups) < 2:
                break

            # 2. Calculate fundamental frequency (minimum stage frequency) for each Vessel ID
            vessel_fundamentals = {}
            for vid, states in vessel_groups.items():
                vessel_fundamentals[vid] = min(s.mean_frequency for s in states)

            # 3. Compare all pairs and merge if they are close or harmonic
            vessels = list(vessel_fundamentals.keys())
            n = len(vessels)
            merged_this_iteration = False

            for i in range(n):
                for j in range(i + 1, n):
                    vidA = vessels[i]
                    vidB = vessels[j]
                    
                    fundA = vessel_fundamentals[vidA]
                    fundB = vessel_fundamentals[vidB]
                    
                    is_harmonic = False
                    harmonic_reason = ""
                    f1 = min(fundA, fundB)
                    f2 = max(fundA, fundB)
                    if 10.0 < f1 <= 150.0 and f2 <= 500.0:
                        ratio = f2 / f1
                        k = round(ratio)
                        if 2 <= k <= 8:
                            expected_f = k * f1
                            if abs(f2 - expected_f) <= self.consolidation_threshold_hz:
                                is_harmonic = True
                                harmonic_reason = f"harmonic relationship of fundamentals (k={k}, {f2:.1f}Hz is harmonic of fundamental {f1:.1f}Hz)"

                    if abs(fundA - fundB) <= self.consolidation_threshold_hz or is_harmonic:
                        # Determine which ID to keep (older one, i.e., lower index number)
                        try:
                            numA = int(vidA.split()[-1])
                            numB = int(vidB.split()[-1])
                            keep_id = vidA if numA < numB else vidB
                            discard_id = vidB if numA < numB else vidA
                        except Exception:
                            keep_id = vidA
                            discard_id = vidB
                        
                        # Rename all states in self.states
                        merge_count = 0
                        for s in self.states:
                            if s.vessel_id == discard_id:
                                s.vessel_id = keep_id
                                merge_count += 1
                                
                        # Rename active states
                        for s in self.active_states.values():
                            if s.vessel_id == discard_id:
                                s.vessel_id = keep_id
                                merge_count += 1

                        reason = f"close fundamentals (diff: {abs(fundA - fundB):.1f}Hz)" if not is_harmonic else harmonic_reason
                        print(f"\n>>> [{current_time:.1f}s] CONSOLIDATED: Merged {discard_id} (fund ~{fundB:.1f}Hz) into {keep_id} (fund ~{fundA:.1f}Hz) due to {reason}")
                        merged_this_iteration = True
                        break
                if merged_this_iteration:
                    break
            
            if not merged_this_iteration:
                break

    def update_multi(self, current_time, detections):
        """
        Detections is a list of dicts: 
        [{'component_index': i, 'score': score, 'centroid': centroid, 'spread': spread}, ...]
        """
        # 0. Perform real-time spectral clustering to merge duplicate/harmonic active states
        self.cluster_active_states(current_time)

        # 1. Close active states that have timed out (> 45s of no updates)
        timeout_limit = 45.0
        states_to_close = []
        for active_state in list(self.active_states.values()):
            if current_time - self.last_seen_time[active_state] > timeout_limit:
                states_to_close.append(active_state)
        for state in states_to_close:
            if not getattr(self, 'q_agent', None):
                print(f"\n>>> [{current_time:.1f}s] Active state for {state.vessel_id} at {state.mean_frequency:.1f}Hz timed out.")
            self.close_state(state, current_time)

        # 2. Filter valid detections
        valid_detections = []
        for det in detections:
            if det['centroid'] > 0:
                valid_detections.append(det)

        if not valid_detections:
            return

        # RL-based matching/decision loop!
        if getattr(self, 'q_agent', None) is not None:
            # Detect whether the agent uses continuous states (LinearFAAgent)
            from core.agent import LinearFAAgent as _LinearFAAgent
            _uses_continuous = isinstance(self.q_agent, _LinearFAAgent)

            for det in valid_detections:
                if _uses_continuous:
                    state = self.rl_env.get_continuous_state(det)
                else:
                    state = self.rl_env.get_state(det)
                action = self.q_agent.get_action(state, epsilon=self.rl_epsilon)
                reward, step_info = self.rl_env.step(action, det, current_time)
                if _uses_continuous:
                    next_state = self.rl_env.get_continuous_state(det)
                else:
                    next_state = self.rl_env.get_state(det)
                
                if self.rl_epsilon > 0.0:
                    sig = inspect.signature(self.q_agent.learn)
                    if 'next_action' in sig.parameters:
                        next_action = self.q_agent.get_action(next_state, epsilon=self.rl_epsilon)
                        self.q_agent.learn(state, action, reward, next_state, next_action)
                    else:
                        self.q_agent.learn(state, action, reward, next_state)
                    
                if self.rl_stats is not None:
                    self.rl_stats['total_reward'] += reward
                    self.rl_stats['action_counts'][action] = self.rl_stats['action_counts'].get(action, 0) + 1
                    status = step_info.get("status", "unknown")
                    self.rl_stats['status_counts'][status] = self.rl_stats['status_counts'].get(status, 0) + 1
                
                status = step_info.get("status", "unknown")
                self.rl_history.append({
                    'time': current_time,
                    'freq': det['centroid'],
                    'status': status
                })

            self.consolidate_all_vessels(current_time)
            return

        matched_detections = set() # indices in valid_detections
        matched_active_states = set() # VesselState objects

        # 3. Associate valid detections with active states (Nearest Neighbor in frequency <= 60 Hz)
        active_list = list(self.active_states.values())
        
        for active_state in active_list:
            best_det_idx = None
            best_freq_diff = float('inf')
            
            for det_idx, det in enumerate(valid_detections):
                if det_idx in matched_detections:
                    continue
                freq_diff = abs(det['centroid'] - active_state.mean_frequency)
                if freq_diff <= self.association_threshold_hz and freq_diff < best_freq_diff:
                    best_freq_diff = freq_diff
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                det = valid_detections[best_det_idx]
                active_state.add_observation(det['centroid'], det['spread'], det['amplitude'])
                self.last_seen_time[active_state] = current_time
                matched_detections.add(best_det_idx)
                matched_active_states.add(active_state)

        # 4. Check for Speed Changes (frequency drift <= proximity_threshold_hz) for remaining unmatched active states
        for active_state in active_list:
            if active_state in matched_active_states:
                continue
                
            best_det_idx = None
            best_freq_diff = float('inf')
            
            for det_idx, det in enumerate(valid_detections):
                if det_idx in matched_detections:
                    continue
                
                centroid = det['centroid']
                mean_f = active_state.mean_frequency
                dist_normal = abs(centroid - mean_f)
                
                if dist_normal <= self.proximity_threshold_hz:
                    if dist_normal < best_freq_diff:
                        best_freq_diff = dist_normal
                        best_det_idx = det_idx
                    
            if best_det_idx is not None:
                det = valid_detections[best_det_idx]
                vessel_id = active_state.vessel_id
                self.close_state(active_state, current_time)
                new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vessel_id)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] {vessel_id} Changed Speed! New Freq: {det['centroid']:.1f} Hz")
                matched_detections.add(best_det_idx)
                matched_active_states.add(active_state)

        # 5. Check for Re-acquisition from archived/completed states (ended within 45s, frequency diff <= 60 Hz)
        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_detections:
                continue
                
            best_state = None
            best_freq_diff = float('inf')
            
            for state in self.states:
                if state.end_time and (current_time - state.end_time <= 45.0):
                    freq_diff = abs(det['centroid'] - state.mean_frequency)
                    if freq_diff <= self.association_threshold_hz and freq_diff < best_freq_diff:
                        best_freq_diff = freq_diff
                        best_state = state
                        
            if best_state is not None:
                vid = best_state.vessel_id
                new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vid)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] Re-acquired {vid} at {det['centroid']:.1f} Hz")
                matched_detections.add(det_idx)

        # 7. Spurious new detections -> spawn a new Vessel ID!
        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_detections:
                continue
                
            self.vessel_counter += 1
            vid = f"Vessel {self.vessel_counter}"
            
            new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vid)
            self.active_states[new_state] = new_state
            self.last_seen_time[new_state] = current_time
            print(f"\n>>> [{current_time:.1f}s] New Vessel Detected! Assigned ID: {vid} at {det['centroid']:.1f} Hz")

        # 8. Consolidate different Vessel IDs if their overall cumulative means are near
        self.consolidate_all_vessels(current_time)

    def close_state(self, state, current_time):
        self.active_states.pop(state, None)
        self.last_seen_time.pop(state, None)
        state.close(current_time)
        duration = state.end_time - state.start_time
        if duration >= self.min_duration_sec:
            self.states.append(state)
            if not getattr(self, 'q_agent', None) or self.rl_epsilon <= 0.05:
                print(f">>> [STORED SPEED STATE] {state.vessel_id} | Mean Freq: {state.mean_frequency:.1f} Hz | "
                      f"Interval: {state.start_time:.1f}s - {state.end_time:.1f}s (Duration: {duration:.1f}s)")
