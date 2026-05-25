import asyncio
import warnings

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning



class VesselState:
    def __init__(self, start_time, initial_freq, initial_spread, vessel_id=None):
        self.start_time = start_time
        self.end_time = None
        self.frequencies = [initial_freq]
        self.spreads = [initial_spread]
        self.vessel_id = vessel_id

    def add_observation(self, freq, spread):
        self.frequencies.append(freq)
        self.spreads.append(spread)

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
    def __init__(self, drift_threshold_hz=35.0, min_vessel_score=0.45, min_duration_sec=10.0):
        self.drift_threshold_hz = drift_threshold_hz
        self.min_vessel_score = min_vessel_score
        self.min_duration_sec = min_duration_sec
        
        self.states = []                  # completed/archived states
        self.active_states = {}           # active_state_obj -> active_state_obj
        self.last_seen_time = {}          # active_state_obj -> float (last seen timestamp)
        self.vessel_counter = 0

    @property
    def current_state(self):
        if self.active_states:
            return next(iter(self.active_states.values()))
        return None

    def cluster_active_states(self, current_time):
        """
        Runs a real-time spectral clustering over active states to merge duplicates
        (proximity within 45 Hz) and harmonics (integer multiples) under the same Vessel ID.
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
                    
                meanA = stateA.mean_frequency
                meanB = stateB.mean_frequency
                if meanA == 0 or meanB == 0:
                    continue
                    
                should_merge = False
                reason = ""
                
                # Case 1: Proximity check (same physical vessel split across NMF components)
                if abs(meanA - meanB) <= 45.0:
                    should_merge = True
                    reason = f"frequency proximity ({meanB:.1f}Hz vs {meanA:.1f}Hz)"
                    
                # Case 2: Harmonic relationship check
                else:
                    ratio = meanB / meanA
                    for k in range(2, 9):
                        if abs(ratio - k) <= 0.06 * k:
                            should_merge = True
                            reason = f"Harmonic {k} relationship ({meanB:.0f}Hz is {k}x of {meanA:.0f}Hz)"
                            break
                        if abs(1.0/ratio - k) <= 0.06 * k:
                            should_merge = True
                            reason = f"Fundamental relationship ({meanB:.0f}Hz is 1/{k}th of {meanA:.0f}Hz)"
                            break
                            
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
            print(f"\n>>> [{current_time:.1f}s] Active state for {state.vessel_id} at {state.mean_frequency:.1f}Hz timed out.")
            self.close_state(state, current_time)

        # 2. Filter valid detections
        valid_detections = []
        for det in detections:
            if det['score'] >= self.min_vessel_score and det['centroid'] > 0:
                valid_detections.append(det)

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
                if freq_diff <= 60.0 and freq_diff < best_freq_diff:
                    best_freq_diff = freq_diff
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                det = valid_detections[best_det_idx]
                active_state.add_observation(det['centroid'], det['spread'])
                self.last_seen_time[active_state] = current_time
                matched_detections.add(best_det_idx)
                matched_active_states.add(active_state)

        # 4. Check for Speed Changes (frequency drift <= 120 Hz OR doubling/halving) for remaining unmatched active states
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
                
                # Model the physical expectation of the vessel's speed profiles
                dist_normal = abs(centroid - mean_f)
                dist_doubled = abs(centroid - 2.0 * mean_f)
                dist_halved = abs(centroid - 0.5 * mean_f)
                
                best_match_dist = float('inf')
                is_match = False
                
                if dist_normal <= 120.0:
                    best_match_dist = dist_normal
                    is_match = True
                if dist_doubled <= 150.0:  # Allow 150 Hz margin around the doubled frequency
                    if dist_doubled < best_match_dist:
                        best_match_dist = dist_doubled
                        is_match = True
                if dist_halved <= 80.0:   # Allow 80 Hz margin around halved frequency
                    if dist_halved < best_match_dist:
                        best_match_dist = dist_halved
                        is_match = True
                        
                if is_match and best_match_dist < best_freq_diff:
                    best_freq_diff = best_match_dist
                    best_det_idx = det_idx
                    
            if best_det_idx is not None:
                det = valid_detections[best_det_idx]
                vessel_id = active_state.vessel_id
                # Close old speed state
                self.close_state(active_state, current_time)
                # Start new speed state for the SAME vessel ID
                new_state = VesselState(current_time, det['centroid'], det['spread'], vessel_id=vessel_id)
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
                    if freq_diff <= 60.0 and freq_diff < best_freq_diff:
                        best_freq_diff = freq_diff
                        best_state = state
                        
            if best_state is not None:
                vid = best_state.vessel_id
                # Re-activate vessel
                new_state = VesselState(current_time, det['centroid'], det['spread'], vessel_id=vid)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] Re-acquired {vid} at {det['centroid']:.1f} Hz")
                matched_detections.add(det_idx)

        # 6. Check for Harmonics Association (integer multiples/divisors) with active vessels
        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_detections:
                continue
                
            centroid = det['centroid']
            matched_vessel_id = None
            matched_reason = ""
            
            for active_state in list(self.active_states.values()):
                mean_f = active_state.mean_frequency
                if mean_f == 0 or centroid == 0:
                    continue
                    
                ratio = centroid / mean_f
                # Check harmonics up to the 8th order
                for k in range(2, 9):
                    # det is a harmonic of active_state
                    if abs(ratio - k) <= 0.06 * k:
                        matched_vessel_id = active_state.vessel_id
                        matched_reason = f"Harmonic {k} of fundamental {mean_f:.0f}Hz"
                        break
                    # det is the fundamental, active_state is a harmonic
                    if abs(1.0/ratio - k) <= 0.06 * k:
                        matched_vessel_id = active_state.vessel_id
                        matched_reason = f"Fundamental of harmonic {mean_f:.0f}Hz (1/{k}th)"
                        break
                        
                if matched_vessel_id:
                    break
                    
            if matched_vessel_id:
                new_state = VesselState(current_time, det['centroid'], det['spread'], vessel_id=matched_vessel_id)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] Associated Harmonic target with {matched_vessel_id} at {centroid:.1f} Hz ({matched_reason})")
                matched_detections.add(det_idx)

        # 7. Spurious new detections -> spawn a new Vessel ID!
        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_detections:
                continue
                
            self.vessel_counter += 1
            vid = f"Vessel {self.vessel_counter}"
            
            new_state = VesselState(current_time, det['centroid'], det['spread'], vessel_id=vid)
            self.active_states[new_state] = new_state
            self.last_seen_time[new_state] = current_time
            print(f"\n>>> [{current_time:.1f}s] New Vessel Detected! Assigned ID: {vid} at {det['centroid']:.1f} Hz")

    def close_state(self, state, current_time):
        self.active_states.pop(state, None)
        self.last_seen_time.pop(state, None)
        state.close(current_time)
        duration = state.end_time - state.start_time
        if duration >= self.min_duration_sec:
            self.states.append(state)
            print(f">>> [STORED SPEED STATE] {state.vessel_id} | Mean Freq: {state.mean_frequency:.1f} Hz | "
                  f"Interval: {state.start_time:.1f}s - {state.end_time:.1f}s (Duration: {duration:.1f}s)")


class DispatcherAgent:
    """
    The dispatcher agent definition.
    This agent observes the audio environment, performs analysis, and visualizes the results.
    """

    def __init__(self, env, min_freq, max_freq, n_fft, n_components, n_max_iter, window_sec=15.0):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._env = env
        self.n_fft = n_fft
        self.n_components = n_components
        self.max_iter = n_max_iter
        
        self.window_sec = window_sec
        self.current_time = 0.0
        self.frames_observed = 0

        # NMF Analysis setup
        self._is_fitting_nmf = False
        self.nmf_update_period_sec = 5.0  # Re-fit NMF model less frequently
        self._nmf_model = None

        # Denoising configuration parameters
        self.signal_smoothing_alpha = 0.15
        self.smoothed_magnitude = None
        
        num_bins = len(self._env.freqs_plot)
        self.spec_sub_kernel = min(51, num_bins // 3)
        if self.spec_sub_kernel % 2 == 0:
            self.spec_sub_kernel += 1
        if self.spec_sub_kernel < 3:
            self.spec_sub_kernel = 3
            
        self.temporal_noise_subtraction_factor = 0.5
        self.completed = False

        # Visualization setup
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        
        self.fig, (self.ax_spec, self.ax_nmf, self.ax_track) = plt.subplots(
            3, 1, 
            figsize=(12, 11),
            gridspec_kw={'height_ratios': [3, 2, 2]}
        )
        self.fig.subplots_adjust(hspace=0.6)
        
        # Tracker visualization setup
        self.ax_kde = self.ax_track.twiny()
        self.ax_track.set_xlim(0, 15.0)
        self.ax_track.set_ylim(self.min_freq, self.max_freq)
        self.ax_track.set_xlabel("Absolute Time (seconds)")
        self.ax_track.set_ylabel("Frequency (Hz)")
        self.ax_track.set_title("Vessel Speed States Timeline")
        self.ax_track.grid(True, linestyle="--", alpha=0.5)

        self.ax_kde.set_xlim(0, 1.1)
        self.ax_kde.set_xlabel("Probability Density (KDE)", color="purple")
        self.ax_kde.tick_params(axis='x', colors="purple")
        
        self.xaxis_extent = [-self.window_sec, 0]

        frames_per_sec_display = librosa.time_to_frames(1.0, sr=self._env.sr, hop_length=self._env.fft_hop_length)
        self.frames_per_window = int(self.window_sec * frames_per_sec_display)

        num_freq_bins = len(self._env.freqs_plot)
        self.S_buffer = np.full((num_freq_bins, self.frames_per_window), -80.0)
        self.stft_magnitude_buffer = np.full((num_freq_bins, self.frames_per_window), 1e-6, dtype=np.float64)
        self.activations_buffer = np.zeros((self.n_components, self.frames_per_window), dtype=np.float64)

        extent = [self.xaxis_extent[0], self.xaxis_extent[1], self._env.freqs_plot[0], self._env.freqs_plot[-1]]
        self.img = self.ax_spec.imshow(self.S_buffer, aspect='auto', origin='lower', 
                        cmap='magma', extent=extent, vmin=-80, vmax=0)
        self.ax_spec.set_xlim(self.xaxis_extent)
        self.title = self.ax_spec.set_title("Live Spectrogram")
        self.ax_spec.set_xlabel("Time (s, relative to now)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.fig.colorbar(self.img, ax=self.ax_spec, format="%+2.0f dB")

        self.lines_nmf = []
        nmf_x_axis = self._env.freqs_plot
        for i in range(self.n_components):
            line, = self.ax_nmf.plot(nmf_x_axis, np.zeros(num_freq_bins), label=f'C{i+1}')
            self.lines_nmf.append(line)
        self.ax_nmf.legend(loc='upper right')
        self.ax_nmf.set_title('NMF Components weighted by Current Activation')
        self.ax_nmf.set_xlabel("Frequency (Hz)")
        self.ax_nmf.set_ylabel('Weighted Magnitude')
        self.ax_nmf.set_xlim(self._env.freqs_plot[0], self._env.freqs_plot[-1])
        self.ax_nmf.set_ylim(0, 0.1)

        plt.show(block=False)

    def _to_decibels(self, magnitude_frame):
        return librosa.amplitude_to_db(magnitude_frame, ref=self._env.global_max)

    def _transform_nmf_frame(self, frame_data):
        """Uses the fitted NMF model to predict activations for a single frame."""
        if self._nmf_model is None:
            return np.zeros((self.n_components, 1), dtype=np.float64)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            # transform expects shape (n_samples, n_features) -> (1, n_features)
            # Make sure to convert frame data to float64 to match model components dtype
            W_frame = self._nmf_model.transform(frame_data.T.astype(np.float64)) 
            return W_frame.T

    async def _background_update_nmf_model(self):
        """Periodically updates the NMF model dictionary (H) from scratch and sorts by frequency."""
        if self._is_fitting_nmf:
            return
        
        self._is_fitting_nmf = True
        try:
            data_to_fit = self.stft_magnitude_buffer.T.copy()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                
                freqs = self._env.freqs_plot
                num_freqs = len(freqs)
                
                # Seed H with gaussian bumps to force coverage of all frequencies.
                # We bias the centers quadratically so more components focus on the lower frequencies.
                H_init = np.random.rand(self.n_components, num_freqs) * 0.01 + 0.01
                centers_norm = np.power(np.linspace(0, 1, self.n_components), 2)
                centers = centers_norm * (freqs[-1] - freqs[0]) + freqs[0]
                
                sigma = (freqs[-1] - freqs[0]) / (self.n_components * 3)
                for i in range(self.n_components):
                    bump = np.exp(-0.5 * ((freqs - centers[i]) / sigma) ** 2)
                    H_init[i] += bump
                
                W_init = np.random.rand(data_to_fit.shape[0], self.n_components) * 0.1 + 0.1
                
                # We use Kullback-Leibler divergence because it is far superior for audio spectra.
                # Frobenius norm tends to ignore low frequencies if high frequencies have higher absolute variance.
                model = NMF(
                    n_components=self.n_components, 
                    init='custom',
                    solver='mu',
                    beta_loss='kullback-leibler',
                    max_iter=400, 
                    random_state=42
                )
                W = await asyncio.to_thread(model.fit_transform, data_to_fit, W=W_init, H=H_init)
                H = model.components_
                
                # Normalize H so its max is 1.0 (W will absorb the scale). This ensures
                # that we can always see the peaks and that our 0 to 0.1 absolute y-axis scale works properly.
                norms = np.max(H, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                H = H / norms
                W = W * norms.T
                
                # To solve the "Permutation Problem" and keep components stable,
                # we sort the components by their spectral centroid (low frequencies first).
                centroids = np.sum(H * freqs, axis=1) / (np.sum(H, axis=1) + 1e-9)
                sort_indices = np.argsort(centroids)
                
                # Apply sorting so C1 is ALWAYS the lowest pitch, C2 is next, etc.
                model.components_ = H[sort_indices]
                W_sorted = W[:, sort_indices]
                
                self._nmf_model = model
                self.activations_buffer[:, :] = W_sorted.T

        except Exception as e:
            print(f"NMF background fit failed: {e}")
        finally:
            self._is_fitting_nmf = False

    def _update_plot_data(self, status):
        self.img.set_data(self.S_buffer)
        
        buffer_size, buffer_percentage = status
        self.title.set_text(
            f"Live Spectrogram (T={self.current_time:.2f}s) | "
            f"Buffer: {buffer_size}/{self._env.max_buffered_chunks} Frames ({buffer_percentage:.0f}%)"
        )
        
        if self._nmf_model is not None:
            H = self._nmf_model.components_
            current_activations = self.activations_buffer[:, -1]
            
            for i, line in enumerate(self.lines_nmf):
                line.set_ydata(H[i] * current_activations[i])
        else:
            for line in self.lines_nmf:
                line.set_ydata(np.zeros(len(self._env.freqs_plot)))

    async def _read_observations_loop(self):
        while True:
            observation, status = await self._env.observe()
            if observation is None:
                self.completed = True
                break

            self.frames_observed += 1
            
            # Make sure it's float64 before assigning to buffer
            magnitude_frame = np.abs(observation)[self._env.freq_mask].astype(np.float64)
            magnitude_frame = np.maximum(magnitude_frame, 1e-6)
            
            # --- Three-Stage Signal Denoising Pipeline ---
            # Stage 1: Spectral Median Filtering (estimates and subtracts broadband/diffuse noise floor)
            broadband_noise_floor = median_filter(magnitude_frame, size=self.spec_sub_kernel)
            
            # 1.2x over-subtraction to aggressively clean up ambient ocean noise
            denoised_magnitude = magnitude_frame - 1.2 * broadband_noise_floor
            denoised_magnitude = np.maximum(denoised_magnitude, 0.03 * broadband_noise_floor)
            denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
            
            # Stage 2: Temporal Background Subtraction (removes stationary equipment/grid hums)
            # Only apply if we have enough observations in the buffer for a stable estimate
            if self.frames_observed > 20:
                # Calculate long-term median of each frequency bin over the buffered sliding window
                temporal_background = np.median(self.stft_magnitude_buffer, axis=1)
                
                # Subtract a portion of the stationary background hums
                denoised_magnitude = denoised_magnitude - self.temporal_noise_subtraction_factor * temporal_background
                denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
                
            # Stage 3: Temporal Recursive Smoothing (EMA filter to prevent frame-to-frame jumping)
            if self.smoothed_magnitude is None:
                self.smoothed_magnitude = denoised_magnitude.copy()
            else:
                self.smoothed_magnitude = (self.signal_smoothing_alpha * denoised_magnitude) + \
                                          ((1.0 - self.signal_smoothing_alpha) * self.smoothed_magnitude)
            
            # Save the denoised, smoothed frame to our magnitude buffer
            self.stft_magnitude_buffer = np.roll(self.stft_magnitude_buffer, -1, axis=1)
            self.stft_magnitude_buffer[:, -1] = self.smoothed_magnitude
            
            # Decibel conversion of the denoised & smoothed frame
            frame_db = self._to_decibels(self.smoothed_magnitude)
            self.S_buffer = np.roll(self.S_buffer, -1, axis=1)
            self.S_buffer[:, -1] = frame_db
            
            # Project using NMF model on the denoised frame
            new_activation = await asyncio.to_thread(
                self._transform_nmf_frame, 
                self.smoothed_magnitude.reshape(-1, 1)
            )
            
            self.activations_buffer = np.roll(self.activations_buffer, -1, axis=1)
            self.activations_buffer[:, -1] = new_activation.flatten()

            time_per_frame = self._env.fft_hop_length / self._env.sr
            self.current_time += time_per_frame
            
            await asyncio.sleep(0)

    async def _nmf_analysis_loop(self):
        while True:
            if self.frames_observed > self.frames_per_window:
                await self._background_update_nmf_model()
            
            await asyncio.sleep(self.nmf_update_period_sec)

    async def _plot_update_loop(self):
        while True:
            if not plt.fignum_exists(self.fig.number):
                break
            
            status = await self._env.get_buffer_status()
            self._update_plot_data(status)
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            await asyncio.sleep(1/15)

    async def start(self):
        asyncio.create_task(self._read_observations_loop())
        asyncio.create_task(self._nmf_analysis_loop())
        asyncio.create_task(self._plot_update_loop())


class SignalProcessorAgent:
    def __init__(self, dispatcher_agent):
        self._dispatcher = dispatcher_agent
        self.n_components = self._dispatcher.n_components
        self._analysis_interval_sec = 1.0
        
        # --- Analysis Results ---
        self.vessel_scores = np.zeros(self.n_components)
        self.feature_scores = {
            'low_freq': np.zeros(self.n_components),
            'mid_freq': np.zeros(self.n_components),
            'tonality': np.zeros(self.n_components),
            'stability': np.zeros(self.n_components),
        }
        self.peak_centroids = np.zeros(self.n_components)
        self.peak_spreads = np.zeros(self.n_components)
        
        self._annotations = []
        for i in range(self.n_components):
            ax = self._dispatcher.ax_nmf
            annotation = ax.text(0, 0, '', 
                                 transform=ax.transAxes, 
                                 fontsize=6, 
                                 color='white',
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
            self._annotations.append(annotation)

        # Vessel Speed Tracking
        self.tracker = VesselStateTracker(drift_threshold_hz=15.0, min_vessel_score=0.45)
        self.smoothed_centroids = np.zeros(self.n_components)
        self.smoothed_spreads = np.zeros(self.n_components)

    def _analyze_component_peaks(self, component_index, H_component, freqs):
        """Finds peaks in an NMF component and calculates their statistics around the dominant tonal peak."""
        max_h = np.max(H_component)
        if max_h == 0:
            self.peak_centroids[component_index] = 0
            self.peak_spreads[component_index] = 0
            return

        # Find significant peaks
        peaks, _ = find_peaks(H_component, height=max_h * 0.05, prominence=max_h * 0.02)
        
        if len(peaks) > 0:
            # 1. Isolate the single tallest peak (dominant tonal)
            tallest_idx = np.argmax(H_component[peaks])
            dom_peak = peaks[tallest_idx]
            
            # 2. Measure spread strictly in a local window of 7 bins around the dominant peak
            local_window = slice(max(0, dom_peak - 3), min(len(H_component), dom_peak + 4))
            local_freqs = freqs[local_window]
            local_weights = H_component[local_window]
            
            local_weight_sum = np.sum(local_weights)
            if local_weight_sum > 0:
                local_centroid = np.sum(local_freqs * local_weights) / local_weight_sum
                local_variance = np.sum((local_freqs - local_centroid)**2 * local_weights) / local_weight_sum
                
                self.peak_centroids[component_index] = local_centroid
                self.peak_spreads[component_index] = np.sqrt(local_variance)
            else:
                self.peak_centroids[component_index] = freqs[dom_peak]
                self.peak_spreads[component_index] = 0.0
        else:
            # If no peaks are found, reset the stats
            self.peak_centroids[component_index] = 0
            self.peak_spreads[component_index] = 0

    async def _analysis_loop(self):
        """Periodically calculates Vessel Score and peak stats for each component."""
        while True:
            nmf_model = self._dispatcher._nmf_model
            if nmf_model is not None:
                H = nmf_model.components_
                W = self._dispatcher.activations_buffer
                freqs = self._dispatcher._env.freqs_plot
                
                low_freq_mask = freqs <= 400
                mid_freq_mask = (freqs > 400) & (freqs <= 1000)
                
                for i in range(self.n_components):
                    h_i = H[i]
                    w_i = W[i]
                    
                    # --- Feature Calculation ---
                    energy_total = np.sum(h_i)
                    if energy_total == 0: continue

                    low_freq_score = np.sum(h_i[low_freq_mask]) / energy_total
                    mid_freq_score = np.sum(h_i[mid_freq_mask]) / energy_total
                    
                    amean = np.mean(h_i)
                    tonality_score = 1.0 - (np.exp(np.mean(np.log(h_i + 1e-9))) / amean) if amean > 0 else 0
                    
                    std_of_diffs = np.std(np.diff(w_i))
                    stability_score = 1.0 / (1.0 + std_of_diffs)

                    # --- Store Feature Scores ---
                    self.feature_scores['low_freq'][i] = low_freq_score
                    self.feature_scores['mid_freq'][i] = mid_freq_score
                    self.feature_scores['tonality'][i] = tonality_score
                    self.feature_scores['stability'][i] = stability_score
                    
                    # --- Final Weighted Vessel Score ---
                    self.vessel_scores[i] = (low_freq_score * 0.3) + (mid_freq_score * 0.4) + (tonality_score * 0.2) + (stability_score * 0.1)
                    
                    # --- Peak Analysis ---
                    self._analyze_component_peaks(i, h_i, freqs)

                # Track all components that meet the vessel score criteria
                if len(self.vessel_scores) > 0:
                    detections = []
                    for i in range(self.n_components):
                        score = self.vessel_scores[i]
                        raw_centroid = self.peak_centroids[i]
                        raw_spread = self.peak_spreads[i]
                        
                        # Apply exponential moving average (smoothing) to reduce tracking jitter
                        if raw_centroid > 0:
                            alpha = 0.25  # Smoothing factor
                            if self.smoothed_centroids[i] == 0:
                                self.smoothed_centroids[i] = raw_centroid
                                self.smoothed_spreads[i] = raw_spread
                            else:
                                self.smoothed_centroids[i] = (alpha * raw_centroid) + ((1.0 - alpha) * self.smoothed_centroids[i])
                                self.smoothed_spreads[i] = (alpha * raw_spread) + ((1.0 - alpha) * self.smoothed_spreads[i])
                                
                            centroid = self.smoothed_centroids[i]
                            spread = self.smoothed_spreads[i]
                        else:
                            centroid = 0
                            spread = 0
                            
                        detections.append({
                            'component_index': i,
                            'score': score,
                            'centroid': centroid,
                            'spread': spread
                        })
                    
                    self.tracker.update_multi(
                        current_time=self._dispatcher.current_time,
                        detections=detections
                    )
            
            await asyncio.sleep(self._analysis_interval_sec)
            
    def _update_annotations(self):
        """Updates plot annotations with scores and peak stats."""
        n_cols = 3
        for i, annotation in enumerate(self._annotations):
            score = self.vessel_scores[i]
            centroid = self.peak_centroids[i]
            spread = self.peak_spreads[i]
            
            lf = self.feature_scores['low_freq'][i]
            mf = self.feature_scores['mid_freq'][i]
            t = self.feature_scores['tonality'][i]
            s = self.feature_scores['stability'][i]
            
            x_pos = 0.02 + (i % n_cols) * 0.33
            y_pos = 0.98 - (i // n_cols) * 0.25
            
            annotation.set_position((x_pos, y_pos))
            annotation.set_text(
                f'C{i+1}: S={score:.2f} μ={centroid:.0f} σ={spread:.0f}\n'
                f'L:{lf:.2f} M:{mf:.2f} T:{t:.2f} S:{s:.2f}'
            )
            
            self._dispatcher.lines_nmf[i].set_color(plt.cm.coolwarm(score))

    def _update_tracker_plot(self):
        """Draws the speed timeline and the probability density curve on ax_track and ax_kde."""
        if not hasattr(self._dispatcher, 'ax_track') or not plt.fignum_exists(self._dispatcher.fig.number):
            return
            
        # 1. Clear both axes
        self._dispatcher.ax_track.clear()
        self._dispatcher.ax_kde.clear()

        # Re-enable grids and limits for ax_track since we cleared it
        self._dispatcher.ax_track.grid(True, linestyle="--", alpha=0.5)
        self._dispatcher.ax_track.set_ylim(self._dispatcher.min_freq, self._dispatcher.max_freq)
        self._dispatcher.ax_track.set_xlabel("Absolute Time (seconds)")
        self._dispatcher.ax_track.set_ylabel("Frequency (Hz)")
        self._dispatcher.ax_track.set_title("Vessel Speed States Timeline")

        # 2. Get all states (archived + active if stable)
        all_states = list(self.tracker.states)
        for active_state in self.tracker.active_states.values():
            active_duration = self._dispatcher.current_time - active_state.start_time
            if active_duration >= self.tracker.min_duration_sec:
                all_states.append(active_state)

        # 3. Group states by their Vessel ID
        vessel_groups = {}
        for state in all_states:
            vid = state.vessel_id if state.vessel_id else "Noise"
            if vid not in vessel_groups:
                vessel_groups[vid] = []
            vessel_groups[vid].append(state)

        # 4. Calculate total cumulative tracking duration for each Vessel ID
        vessel_durations = []
        for vid, states in vessel_groups.items():
            total_duration = 0.0
            for s in states:
                end_t = s.end_time if s.end_time else self._dispatcher.current_time
                total_duration += (end_t - s.start_time)
            vessel_durations.append((vid, total_duration, states))

        # Sort vessels by cumulative duration (most dominant vessels first)
        vessel_durations.sort(key=lambda x: x[1], reverse=True)

        # We keep the top 8 most dominant vessels (supporting more simultaneous clusters)
        max_vessels = 8
        dominant_vessels = vessel_durations[:max_vessels]
        minor_vessels = vessel_durations[max_vessels:]

        # Map each vessel group to its metadata
        vessel_metadata = {}
        for rank_idx, (vid, dur, states) in enumerate(dominant_vessels):
            # Calculate overall weighted mean frequency for this vessel across all its speed levels
            freqs = [s.mean_frequency for s in states]
            weights = [(s.end_time or self._dispatcher.current_time) - s.start_time for s in states]
            total_w = sum(weights) if sum(weights) > 0 else 1.0
            overall_mean_f = sum(f * w for f, w in zip(freqs, weights)) / total_w
            
            vessel_metadata[vid] = {
                'is_dominant': True,
                'color_idx': rank_idx,
                'label': f"{vid} (~{overall_mean_f:.0f}Hz)",
                'states': states
            }

        for vid, dur, states in minor_vessels:
            vessel_metadata[vid] = {
                'is_dominant': False,
                'color_idx': -1,
                'label': "Minor Noise / Interference",
                'states': states
            }

        # 5. Draw each segment and draw connecting transition guidelines
        cmap = plt.cm.get_cmap("tab10")
        labeled_vessels = set()
        
        total_time = max(15.0, self._dispatcher.current_time)
        # Only label segments that are at least 1.5% of total session time
        label_threshold_sec = max(15.0, 0.015 * total_time) 

        for vid, meta in vessel_metadata.items():
            color = cmap(meta['color_idx'] % 10) if meta['is_dominant'] else "lightgrey"
            linestyle_segment = "--" if any(s in self.tracker.active_states.values() for s in meta['states']) else "-"
            
            # Sort segments of this vessel chronologically to draw transitions
            sorted_segs = sorted(meta['states'], key=lambda s: s.start_time)
            
            # Draw segment lines and shaded variance
            for j, state in enumerate(sorted_segs):
                is_active = (state in self.tracker.active_states.values())
                start_t = state.start_time
                end_t = state.end_time if state.end_time else self._dispatcher.current_time
                duration = end_t - start_t
                
                mean_f = state.mean_frequency
                std_f = np.sqrt(state.total_variance)
                
                if meta['is_dominant']:
                    linestyle = "--" if is_active else "-"
                    linewidth = 3.5
                    alpha_val = 0.95
                    
                    # Shaded variance band for dominant vessels
                    self._dispatcher.ax_track.fill_between(
                        [start_t, end_t], 
                        [mean_f - std_f, mean_f - std_f], 
                        [mean_f + std_f, mean_f + std_f],
                        color=color, alpha=0.15
                    )
                    
                    # Single legend entry per dominant vessel
                    vessel_label = meta['label']
                    legend_label = vessel_label if vessel_label not in labeled_vessels else ""
                    labeled_vessels.add(vessel_label)
                    
                    # Plot segment bar
                    self._dispatcher.ax_track.plot(
                        [start_t, end_t], [mean_f, mean_f], 
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha_val,
                        label=legend_label
                    )
                    
                    # Draw text label only on long, stable segments to prevent overlap clutter
                    if duration >= label_threshold_sec:
                        self._dispatcher.ax_track.text(
                            (start_t + end_t) / 2, mean_f + std_f + 5.0, 
                            f"{mean_f:.0f}Hz ± {std_f:.0f}",
                            color=color, fontsize=8, ha="center", va="bottom",
                            weight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.85, ec=color, lw=0.5)
                        )
                else:
                    # Minor/transient noise: draw as a thin, faded grey background line
                    self._dispatcher.ax_track.plot(
                        [start_t, end_t], [mean_f, mean_f], 
                        color=color, linestyle="-", linewidth=1.0, alpha=0.25
                    )

            # Draw dotted transition guidelines strictly between dominant vessel segments
            if meta['is_dominant'] and len(sorted_segs) > 1:
                for j in range(len(sorted_segs) - 1):
                    segA = sorted_segs[j]
                    segB = sorted_segs[j+1]
                    
                    endA_t = segA.end_time if segA.end_time else self._dispatcher.current_time
                    startB_t = segB.start_time
                    
                    meanA_f = segA.mean_frequency
                    meanB_f = segB.mean_frequency
                    
                    # Connect transition guide
                    self._dispatcher.ax_track.plot(
                        [endA_t, startB_t], [meanA_f, meanB_f],
                        color=color, linestyle=":", linewidth=2.0, alpha=0.7
                    )

        # Set x limits on timeline to absolute time
        self._dispatcher.ax_track.set_xlim(0, max(15.0, self._dispatcher.current_time))
        if all_states:
            self._dispatcher.ax_track.legend(loc="upper left", fontsize=8)

        # 6. Draw the probability density curve on ax_kde
        freq_axis = self._dispatcher._env.freqs_plot
        pdf = np.zeros_like(freq_axis, dtype=np.float64)
        
        # Calculate log-duration weighted PDF using all stable vessel states
        # This ensures all distinct speed stages (clusters) display visible, clear peaks
        if all_states:
            for state in all_states:
                duration = (state.end_time or self._dispatcher.current_time) - state.start_time
                if duration <= 0:
                    duration = 0.5
                
                mean = state.mean_frequency
                variance = max(state.total_variance, 4.0)
                
                # Gaussian kernel peak (normalized to peak height = 1.0)
                exponent = -0.5 * ((freq_axis - mean) ** 2) / variance
                density = np.exp(exponent)
                
                # Weight by log-duration so short stable states are not suppressed by long ones
                weight = np.log1p(duration)
                pdf += weight * density
                
            max_pdf = np.max(pdf)
            if max_pdf > 0:
                pdf = pdf / max_pdf

        # Plot PDF vertically on ax_kde (x = probability, y = frequency)
        self._dispatcher.ax_kde.plot(pdf, freq_axis, color="purple", linewidth=1.5, alpha=0.8, label="Freq Probability")
        self._dispatcher.ax_kde.fill_betweenx(freq_axis, 0, pdf, color="purple", alpha=0.15)
        
        # Configure ax_kde
        self._dispatcher.ax_kde.set_xlim(0, 1.1)
        self._dispatcher.ax_kde.set_xlabel("Probability Density (KDE)", color="purple")
        self._dispatcher.ax_kde.tick_params(axis='x', colors="purple")

    async def _plot_update_loop(self):
        while True:
            if not plt.fignum_exists(self._dispatcher.fig.number):
                break
            
            self._update_annotations()
            try:
                self._update_tracker_plot()
            except Exception as e:
                print(f"Error updating tracker plot: {e}")
                
            await asyncio.sleep(self._analysis_interval_sec)

    async def start(self):
        asyncio.create_task(self._analysis_loop())
        asyncio.create_task(self._plot_update_loop())
