import asyncio
import warnings
import inspect
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ActiveStatesDict(dict):
    """
    Custom dictionary wrapper for active_states.
    Intercepts mutations to dynamically spawn or terminate SignalProcessorAgent child instances.
    """
    def __init__(self, dispatcher):
        super().__init__()
        self.dispatcher = dispatcher

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Track vessel fundamental frequency in cache
        vid = key.vessel_id
        if vid and vid != "Noise" and len(key.frequencies) > 0:
            mean_f = key.frequencies[0]
            cache = self.dispatcher.vessel_fundamentals
            if vid not in cache or mean_f < cache[vid]:
                cache[vid] = mean_f

        # Spawn the child SignalProcessorAgent when a new active state is added
        exists = any(p.vessel_state == key for p in self.dispatcher.active_processors)
        if not exists:
            from core.agent.signal_processor_agent import SignalProcessorAgent
            proc = SignalProcessorAgent(
                env=self.dispatcher,
                start_time=key.start_time,
                initial_freq=key.frequencies[0],
                initial_spread=key.spreads[0],
                initial_amp=key.amplitudes[0],
                vessel_id=key.vessel_id
            )
            # Link the newly created key to this child processor
            proc.vessel_state = key
            self.dispatcher.active_processors.append(proc)

    def pop(self, key, default=None):
        # Terminate and remove the child processor when the active state is removed
        proc_to_remove = None
        for p in self.dispatcher.active_processors:
            if p.vessel_state == key:
                proc_to_remove = p
                break
        if proc_to_remove:
            self.dispatcher.active_processors.remove(proc_to_remove)
            proc_to_remove.terminate(self.dispatcher.current_time)
        return super().pop(key, default)


class DispatcherAgent:
    """
    The dispatcher agent definition.
    This agent acts as the parent environment for the multi-agent vessel tracker.
    It observes the audio environment, performs analysis, and manages child SignalProcessorAgent instances.
    """
    def __init__(self, env, min_freq, max_freq, n_fft, n_components, n_max_iter, window_sec=15.0, headless=False,
                 proximity_threshold_hz=65.0, association_threshold_hz=80.0, peak_spread_window_bins=15,
                 variance_multiplier=1.5, min_variance_floor=25.0, consolidation_threshold_hz=65.0,
                 min_duration_sec=45.0, min_vessel_score=0.50):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._env = env
        self.n_fft = n_fft
        self.n_components = n_components
        self.max_iter = n_max_iter
        self.headless = headless
        
        self.window_sec = window_sec
        self.current_time = getattr(self._env, "start_timestamp", 0.0)
        self.frames_observed = 0

        # NMF Analysis setup
        self._is_fitting_nmf = False
        self.nmf_update_period_sec = 5.0
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

        # --- Analysis Thresholds & Settings ---
        self.proximity_threshold_hz = proximity_threshold_hz
        self.association_threshold_hz = association_threshold_hz
        self.peak_spread_window_bins = peak_spread_window_bins
        self.variance_multiplier = variance_multiplier
        self.min_variance_floor = min_variance_floor
        self.consolidation_threshold_hz = consolidation_threshold_hz
        self.min_duration_sec = min_duration_sec
        self.min_vessel_score = min_vessel_score
        self.drift_threshold_hz = 15.0

        # --- Analysis Results ---
        self.vessel_scores = np.zeros(self.n_components)
        self.feature_scores = {
            'low_freq': np.zeros(self.n_components),
            'mid_freq': np.zeros(self.n_components),
            'high_freq': np.zeros(self.n_components),
            'tonality': np.zeros(self.n_components),
            'stability': np.zeros(self.n_components),
        }
        self.peak_centroids = np.zeros(self.n_components)
        self.peak_spreads = np.zeros(self.n_components)
        self.smoothed_centroids = np.zeros(self.n_components)
        self.smoothed_spreads = np.zeros(self.n_components)

        # --- Multi-agent registries & compatibility interfaces ---
        self.active_processors = []
        self.vessel_fundamentals = {}
        self.active_states = ActiveStatesDict(self)
        self.states = []
        self.last_seen_time = {}
        self.vessel_counter = 0
        self.tracker = self

        # RL policy attributes
        self.rl_agent = None
        self.rl_env = None
        self.rl_epsilon = 0.0
        self.rl_stats = None
        self.rl_history = []

        # Visualization setup
        self._setup_plot()

    @property
    def current_state(self):
        if self.active_processors:
            return self.active_processors[0].vessel_state
        return None

    def _setup_plot(self):
        if self.headless:
            frames_per_sec_display = librosa.time_to_frames(1.0, sr=self._env.sr, hop_length=self._env.fft_hop_length)
            self.frames_per_window = int(self.window_sec * frames_per_sec_display)
            num_freq_bins = len(self._env.freqs_plot)
            self.S_buffer = np.full((num_freq_bins, self.frames_per_window), -80.0)
            self.stft_magnitude_buffer = np.full((num_freq_bins, self.frames_per_window), 1e-6, dtype=np.float64)
            self.activations_buffer = np.zeros((self.n_components, self.frames_per_window), dtype=np.float64)
            self.lines_nmf = []
            return

        plt.ion()
        
        self.fig, (self.ax_spec, self.ax_nmf, self.ax_track, self.ax_amp) = plt.subplots(
            4, 1, 
            figsize=(12, 14),
            gridspec_kw={'height_ratios': [3, 2, 2, 2]}
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

        # Amplitude visualization setup
        self.ax_amp.set_xlim(0, 15.0)
        self.ax_amp.set_ylim(0, 1.0)
        self.ax_amp.set_xlabel("Absolute Time (seconds)")
        self.ax_amp.set_ylabel("Signal Amplitude (Activation)")
        self.ax_amp.set_title("Vessel Signal Amplitude Timeline")
        self.ax_amp.grid(True, linestyle="--", alpha=0.5)
        
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
        self.nmf_ylim_max = 0.1
        self.ax_nmf.set_ylim(0, self.nmf_ylim_max)

        self._annotations = []
        for i in range(self.n_components):
            ax = self.ax_nmf
            annotation = ax.text(0, 0, '', 
                                 transform=ax.transAxes, 
                                 fontsize=6, 
                                 color='white',
                                 verticalalignment='top',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
            self._annotations.append(annotation)

        plt.show(block=False)

    def _to_decibels(self, magnitude_frame):
        return librosa.amplitude_to_db(magnitude_frame, ref=self._env.global_max)

    def _transform_nmf_frame(self, frame_data):
        """Uses the NMF components to project activations for a single frame using fast numpy KL-updates."""
        if self._nmf_model is None:
            return np.zeros((self.n_components, 1), dtype=np.float64)
            
        H = self._nmf_model.components_
        x = frame_data.flatten().astype(np.float64)
        
        w = np.full(self.n_components, 0.1, dtype=np.float64)
        eps = 1e-9
        H_sum = np.sum(H, axis=1)
        H_sum = np.maximum(H_sum, eps)
        
        for _ in range(25):
            recon = w @ H + eps
            ratio = x / recon
            grad = ratio @ H.T
            w = w * (grad / H_sum)
            w = np.maximum(w, 0.0)
            
        return w.reshape(-1, 1)

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
                
                np.random.seed(42)
                H_init = np.random.rand(self.n_components, num_freqs) * 0.01 + 0.01
                centers_norm = np.power(np.linspace(0, 1, self.n_components), 2)
                centers = centers_norm * (freqs[-1] - freqs[0]) + freqs[0]
                
                sigma = (freqs[-1] - freqs[0]) / (self.n_components * 3)
                for i in range(self.n_components):
                    bump = np.exp(-0.5 * ((freqs - centers[i]) / sigma) ** 2)
                    H_init[i] += bump
                
                W_init = np.random.rand(data_to_fit.shape[0], self.n_components) * 0.1 + 0.1
                
                model = NMF(
                    n_components=self.n_components, 
                    init='custom',
                    solver='mu',
                    beta_loss='kullback-leibler',
                    max_iter=40, 
                    random_state=42
                )
                W = await asyncio.to_thread(model.fit_transform, data_to_fit, W=W_init, H=H_init)
                H = model.components_
                
                norms = np.max(H, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                H = H / norms
                W = W * norms.T
                
                centroids = np.sum(H * freqs, axis=1) / np.maximum(np.sum(H, axis=1), 1e-9)
                sort_idx = np.argsort(centroids)
                model.components_ = H[sort_idx]
                W_sorted = W[:, sort_idx]
                
                self._nmf_model = model
                self.activations_buffer = W_sorted.T.copy()
                
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
            
            max_val = 0.0
            for i, line in enumerate(self.lines_nmf):
                ydata = H[i] * current_activations[i]
                line.set_ydata(ydata)
                max_val = max(max_val, np.max(ydata))
            
            target_max = max(max_val * 1.15, 0.01)
            self.nmf_ylim_max = 0.9 * self.nmf_ylim_max + 0.1 * target_max
            self.ax_nmf.set_ylim(0, self.nmf_ylim_max)
        else:
            for line in self.lines_nmf:
                line.set_ydata(np.zeros(len(self._env.freqs_plot)))

    def _analyze_component_peaks(self, component_index, H_component, freqs):
        """Finds peaks in an NMF component and calculates their statistics around the dominant tonal peak."""
        max_h = np.max(H_component)
        if max_h == 0:
            self.peak_centroids[component_index] = 0
            self.peak_spreads[component_index] = 0
            return
 
        peaks, _ = find_peaks(H_component, height=max_h * 0.05, prominence=max_h * 0.02)
        
        if len(peaks) > 0:
            dom_peak = peaks[0]
            
            half_window = self.peak_spread_window_bins // 2
            local_window = slice(max(0, dom_peak - half_window), min(len(H_component), dom_peak + half_window + 1))
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
            self.peak_centroids[component_index] = 0
            self.peak_spreads[component_index] = 0

    def get_vessel_fundamental(self, vessel_id):
        if not vessel_id or vessel_id == "Noise":
            return None
        return self.vessel_fundamentals.get(vessel_id)

    def cluster_active_states(self):
        """
        Runs a real-time spectral clustering over active states to merge duplicates
        and harmonics under the same Vessel ID, based on their fundamental frequencies.
        """
        active_list = list(self.active_processors)
        n = len(active_list)
        if n < 2:
            return
            
        merged_any = False
        
        for i in range(n):
            for j in range(i + 1, n):
                procA = active_list[i]
                procB = active_list[j]
                stateA = procA.vessel_state
                stateB = procB.vessel_state
                
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
                    procA.vessel_id = keep_id
                    procB.vessel_id = keep_id
                    
                    # Update cache
                    fund_keep = self.vessel_fundamentals.get(keep_id, float('inf'))
                    fund_discard = self.vessel_fundamentals.pop(discard_id, float('inf'))
                    self.vessel_fundamentals[keep_id] = min(fund_keep, fund_discard)
                    
                    # Update all historical completed states
                    merge_count = 0
                    for s in self.states:
                        if s.vessel_id == discard_id:
                            s.vessel_id = keep_id
                            merge_count += 1
                            
                    print(f"\n>>> [{self.current_time:.1f}s] Merged {discard_id} into {keep_id} due to {reason} (Merged {merge_count} archived stages)")
                    merged_any = True
                    break
            if merged_any:
                break

    def consolidate_all_vessels(self):
        """
        Consolidates different Vessel IDs together across all completed and active states
        if their fundamental frequencies are within consolidation_threshold_hz
        OR if they exhibit a harmonic relationship (using a frequency-drift tolerant threshold).
        Runs iteratively until no more merges are possible.
        """
        while True:
            # Dynamically compute vessel fundamentals from current states and active processors
            vessel_groups = {}
            for s in self.states:
                vid = s.vessel_id
                if not vid or vid == "Noise":
                    continue
                vessel_groups.setdefault(vid, []).append(s)
            for p in self.active_processors:
                vid = p.vessel_id
                if not vid or vid == "Noise":
                    continue
                vessel_groups.setdefault(vid, []).append(p.vessel_state)

            if len(vessel_groups) < 2:
                break

            # Calculate fundamental frequency (minimum stage frequency) for each Vessel ID
            vessel_fundamentals = {}
            for vid, segs in vessel_groups.items():
                vessel_fundamentals[vid] = min(s.mean_frequency for s in segs)

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
                        try:
                            numA = int(vidA.split()[-1])
                            numB = int(vidB.split()[-1])
                            keep_id = vidA if numA < numB else vidB
                            discard_id = vidB if numA < numB else vidA
                        except Exception:
                            keep_id = vidA
                            discard_id = vidB
                        
                        merge_count = 0
                        for s in self.states:
                            if s.vessel_id == discard_id:
                                s.vessel_id = keep_id
                                merge_count += 1
                                
                        for p in self.active_processors:
                            if p.vessel_id == discard_id:
                                p.vessel_id = keep_id
                                p.vessel_state.vessel_id = keep_id
                                merge_count += 1

                        # Keep local cache updated for consistency (if accessed externally)
                        fund_keep = self.vessel_fundamentals.get(keep_id, float('inf'))
                        fund_discard = self.vessel_fundamentals.pop(discard_id, float('inf'))
                        self.vessel_fundamentals[keep_id] = min(fund_keep, fund_discard)

                        reason = f"close fundamentals (diff: {abs(fundA - fundB):.1f}Hz)" if not is_harmonic else harmonic_reason
                        print(f"\n>>> [{self.current_time:.1f}s] CONSOLIDATED: Merged {discard_id} (fund ~{fundB:.1f}Hz) into {keep_id} (fund ~{fundA:.1f}Hz) due to {reason}")
                        merged_this_iteration = True
                        break
                if merged_this_iteration:
                    break
            
            if not merged_this_iteration:
                break

    def close_state(self, state, current_time):
        self.active_states.pop(state, None)
        self.last_seen_time.pop(state, None)
        state.close(current_time)
        duration = state.end_time - state.start_time
        if duration >= self.min_duration_sec:
            self.states.append(state)
            if not self.rl_agent or self.rl_epsilon <= 0.05:
                print(f">>> [STORED SPEED STATE] {state.vessel_id} | Mean Freq: {state.mean_frequency:.1f} Hz | "
                      f"Interval: {state.start_time:.1f}s - {state.end_time:.1f}s (Duration: {duration:.1f}s)")

    def update_multi(self, current_time, detections):
        # 0. Perform real-time spectral clustering to merge duplicate/harmonic active states
        if self.frames_observed % 25 == 0:
            self.cluster_active_states()

        # 1. Close active states that have timed out (> 45s of no updates)
        timeout_limit = 45.0
        states_to_close = []
        for active_state in list(self.active_states.values()):
            if current_time - self.last_seen_time[active_state] > timeout_limit:
                states_to_close.append(active_state)
        for state in states_to_close:
            self.close_state(state, self.last_seen_time[state])

        # 2. Filter valid detections
        valid_detections = []
        for det in detections:
            if det['centroid'] > 0 and det['score'] >= self.min_vessel_score and det['amplitude'] >= 0.002:
                valid_detections.append(det)

        if not valid_detections:
            return

        # RL-based matching/decision loop
        if self.rl_agent is not None:
            for det in valid_detections:
                current_state = self.rl_agent.observe(det, self.rl_env)
                
                if hasattr(self, '_prev_rl_transition') and self._prev_rl_transition is not None:
                    p_state, p_action, p_reward = self._prev_rl_transition
                    self.rl_agent.step(p_state, p_action, p_reward, current_state)
                
                action = self.rl_agent.act(current_state)
                reward, step_info = self.rl_env.step(action, det, current_time)
                self._prev_rl_transition = (current_state, action, reward)
                    
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

            if self.frames_observed % 25 == 0:
                self.consolidate_all_vessels()
            return

        # Non-RL heuristic matching
        matched_detections = set()
        matched_active_states = set()

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
                
                self.vessel_counter += 1
                vid = vessel_id
                from core.vessel_state import VesselState
                new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vid)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] {vessel_id} Changed Speed! New Freq: {det['centroid']:.1f} Hz")
                matched_detections.add(best_det_idx)
                matched_active_states.add(active_state)

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
                from core.vessel_state import VesselState
                new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vid)
                self.active_states[new_state] = new_state
                self.last_seen_time[new_state] = current_time
                print(f"\n>>> [{current_time:.1f}s] Re-acquired {vid} at {det['centroid']:.1f} Hz")
                matched_detections.add(det_idx)

        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_detections:
                continue
                
            self.vessel_counter += 1
            vid = f"Vessel {self.vessel_counter}"
            
            from core.vessel_state import VesselState
            new_state = VesselState(current_time, det['centroid'], det['spread'], initial_amp=det['amplitude'], vessel_id=vid)
            self.active_states[new_state] = new_state
            self.last_seen_time[new_state] = current_time
            print(f"\n>>> [{current_time:.1f}s] New Vessel Detected! Assigned ID: {vid} at {det['centroid']:.1f} Hz")

        if self.frames_observed % 25 == 0:
            self.consolidate_all_vessels()

    def finalize_rl_training(self):
        """Finalizes the last pending RL transition at the end of the training episode."""
        if hasattr(self, '_prev_rl_transition') and self._prev_rl_transition is not None and self.rl_agent is not None:
            p_state, p_action, p_reward = self._prev_rl_transition
            self.rl_agent.step(p_state, p_action, p_reward, None)
            self._prev_rl_transition = None

    def _update_annotations(self):
        if self.headless:
            return
        n_cols = 3
        for i, annotation in enumerate(self._annotations):
            score = self.vessel_scores[i]
            centroid = self.peak_centroids[i]
            spread = self.peak_spreads[i]
            
            lf = self.feature_scores['low_freq'][i]
            mf = self.feature_scores['mid_freq'][i]
            hf = self.feature_scores['high_freq'][i]
            t = self.feature_scores['tonality'][i]
            s = self.feature_scores['stability'][i]
            
            x_pos = 0.02 + (i % n_cols) * 0.33
            y_pos = 0.98 - (i // n_cols) * 0.25
            
            annotation.set_position((x_pos, y_pos))
            annotation.set_text(
                f'C{i+1}: S={score:.2f} μ={centroid:.0f} σ={spread:.0f}\n'
                f'L:{lf:.2f} M:{mf:.2f} H:{hf:.2f} T:{t:.2f} S:{s:.2f}'
            )
            
            self.lines_nmf[i].set_color(plt.cm.coolwarm(score))

    def _update_tracker_plot(self):
        if not hasattr(self, 'ax_track') or not plt.fignum_exists(self.fig.number):
            return
            
        self.ax_track.clear()
        self.ax_kde.clear()
        if hasattr(self, 'ax_amp'):
            self.ax_amp.clear()

        self.ax_track.grid(True, linestyle="--", alpha=0.5)
        self.ax_track.set_ylim(self.min_freq, self.max_freq)
        self.ax_track.set_xlabel("Real Time (HH:MM:SS)")
        self.ax_track.set_ylabel("Frequency (Hz)")
        self.ax_track.set_title("Vessel Speed States Timeline")

        all_states = list(self.states)
        for p in self.active_processors:
            active_duration = self.current_time - p.vessel_state.start_time
            if active_duration >= self.min_duration_sec:
                all_states.append(p.vessel_state)

        vessel_groups = {}
        for state in all_states:
            vid = state.vessel_id if state.vessel_id else "Noise"
            if vid not in vessel_groups:
                vessel_groups[vid] = []
            vessel_groups[vid].append(state)

        vessel_durations = []
        for vid, states in vessel_groups.items():
            total_duration = 0.0
            for s in states:
                end_t = s.end_time if s.end_time else self.current_time
                total_duration += (end_t - s.start_time)
            if total_duration >= 180.0:
                vessel_durations.append((vid, total_duration, states))

        vessel_durations.sort(key=lambda x: x[1], reverse=True)

        labeled_vessels = set()
        vessel_kde_data = []

        for rank, (vessel_id, total_dur, states) in enumerate(vessel_durations):
            is_dominant = (rank < 8)
            color = plt.cm.tab10(rank % 10)
            
            meta = {
                'vessel_id': vessel_id,
                'is_dominant': is_dominant,
                'color': color,
                'label': f"{vessel_id} ({total_dur:.1f}s)" if is_dominant else "Noise"
            }

            sorted_segs = sorted(states, key=lambda s: s.start_time)
            
            for state in sorted_segs:
                is_active = (state.end_time is None)
                start_t = state.start_time
                end_t = state.end_time if state.end_time else self.current_time
                duration = end_t - start_t
                
                mean_f = state.mean_frequency
                scaled_variance = max(state.total_variance * self.variance_multiplier, self.min_variance_floor)
                std_f = np.sqrt(scaled_variance)
                
                t_seg = np.linspace(start_t, end_t, len(state.amplitudes)) if len(state.amplitudes) > 1 else np.array([start_t])
                
                if meta['is_dominant']:
                    linestyle = "--" if is_active else "-"
                    linewidth = 3.5
                    alpha_val = 0.95
                    
                    self.ax_track.fill_between(
                        [start_t, end_t], 
                        [mean_f - std_f, mean_f - std_f], 
                        [mean_f + std_f, mean_f + std_f],
                        color=color, alpha=0.15
                    )
                    
                    vessel_label = meta['label']
                    legend_label = vessel_label if vessel_label not in labeled_vessels else ""
                    labeled_vessels.add(vessel_label)
                    
                    self.ax_track.plot(
                        [start_t, end_t], [mean_f, mean_f], 
                        color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha_val,
                        label=legend_label
                    )
                    
                    if hasattr(self, 'ax_amp'):
                        self.ax_amp.plot(
                            t_seg, state.amplitudes,
                            color=color, linestyle=linestyle, linewidth=2.0, alpha=alpha_val
                        )
                        
                    for freq_val in state.frequencies:
                        vessel_kde_data.append((freq_val, color))
                else:
                    self.ax_track.plot(
                        [start_t, end_t], [mean_f, mean_f], 
                        color=color, linestyle="-", linewidth=1.0, alpha=0.25
                    )
                    if hasattr(self, 'ax_amp'):
                        self.ax_amp.plot(
                            t_seg, state.amplitudes,
                            color=color, linestyle="-", linewidth=1.0, alpha=0.25
                        )

            if meta['is_dominant'] and len(sorted_segs) > 1:
                for j in range(len(sorted_segs) - 1):
                    segA = sorted_segs[j]
                    segB = sorted_segs[j+1]
                    
                    endA_t = segA.end_time if segA.end_time else self.current_time
                    startB_t = segB.start_time
                    
                    meanA_f = segA.mean_frequency
                    meanB_f = segB.mean_frequency
                    
                    self.ax_track.plot(
                        [endA_t, startB_t], [meanA_f, meanB_f],
                        color=color, linestyle=":", linewidth=2.0, alpha=0.7
                    )

        self.ax_track.axvline(self.current_time, color="red", linestyle="--", linewidth=1.0)
        if hasattr(self, 'ax_amp'):
            self.ax_amp.axvline(self.current_time, color="red", linestyle="--", linewidth=1.0)

        t_start = getattr(self._env, "start_timestamp", 0.0)
        t_limit = max(t_start + 15.0, self.current_time + 1.0)
        self.ax_track.set_xlim(t_start, t_limit)
        if hasattr(self, 'ax_amp'):
            self.ax_amp.set_xlim(t_start, t_limit)

        import datetime
        from matplotlib.ticker import FuncFormatter
        def time_formatter(x, pos):
            try:
                dt = datetime.datetime.fromtimestamp(x)
                return dt.strftime("%H:%M:%S")
            except Exception:
                return f"{x:.1f}"
        self.ax_track.xaxis.set_major_formatter(FuncFormatter(time_formatter))
        if hasattr(self, 'ax_amp'):
            self.ax_amp.xaxis.set_major_formatter(FuncFormatter(time_formatter))

        # Plot RL decisions
        if hasattr(self, 'rl_history') and self.rl_history:
            times_ok, freqs_ok = [], []
            times_fn, freqs_fn = [], []
            times_dup, freqs_dup = [], []
            times_bad, freqs_bad = [], []
            
            for pt in self.rl_history:
                status = pt['status']
                t = pt['time']
                f = pt['freq']
                if status in ("good_association", "correct_spawn"):
                    times_ok.append(t)
                    freqs_ok.append(f)
                elif status == "false_negative":
                    times_fn.append(t)
                    freqs_fn.append(f)
                elif status == "duplicate_spawn_penalty":
                    times_dup.append(t)
                    freqs_dup.append(f)
                elif status in ("bad_association_mismatch", "invalid_association_no_vessels"):
                    times_bad.append(t)
                    freqs_bad.append(f)
            
            is_long_run = self.current_time > 600
            
            if times_ok:
                if is_long_run:
                    # Sparse correct detections to keep the main vessel tracking lines visible
                    self.ax_track.scatter(times_ok[::50], freqs_ok[::50], color="green", s=10, alpha=0.3, marker="o", label="Correct Detection (Sparse)")
                else:
                    self.ax_track.scatter(times_ok, freqs_ok, color="green", s=25, alpha=0.7, marker="o", label="Correct Detection")
            
            if times_fn:
                if is_long_run:
                    # Sparse false negatives
                    self.ax_track.scatter(times_fn[::5], freqs_fn[::5], color="red", s=15, alpha=0.6, marker="x", label="False Negative (Sparse)")
                else:
                    self.ax_track.scatter(times_fn, freqs_fn, color="red", s=35, alpha=0.9, marker="x", label="False Negative (Miss)")
            
            if times_dup:
                if is_long_run:
                    # Sparse duplicate spawns
                    self.ax_track.scatter(times_dup[::10], freqs_dup[::10], color="orange", s=15, alpha=0.6, marker="^", label="Duplicate Spawn (Sparse)")
                else:
                    self.ax_track.scatter(times_dup, freqs_dup, color="orange", s=35, alpha=0.9, marker="^", label="Duplicate Spawn")
            
            if times_bad:
                if is_long_run:
                    # Sparse bad associations
                    self.ax_track.scatter(times_bad[::5], freqs_bad[::5], color="magenta", s=15, alpha=0.6, marker="s", label="Bad Association (Sparse)")
                else:
                    self.ax_track.scatter(times_bad, freqs_bad, color="magenta", s=35, alpha=0.9, marker="s", label="Bad Association")

        self.ax_track.legend(loc="upper left", fontsize=7, framealpha=0.6)

        if len(vessel_kde_data) > 3:
            try:
                from scipy.stats import gaussian_kde
                freq_vals = [pt[0] for pt in vessel_kde_data]
                kde = gaussian_kde(freq_vals, bw_method=0.1)
                
                y_grid = np.linspace(self.min_freq, self.max_freq, 500)
                density = kde(y_grid)
                
                max_d = np.max(density)
                if max_d > 0:
                    density = (density / max_d) * 0.8
                    
                self.ax_kde.plot(density, y_grid, color="purple", linewidth=1.5, alpha=0.7)
                self.ax_kde.fill_betweenx(y_grid, 0, density, color="purple", alpha=0.1)
            except Exception as e:
                print(f"Error drawing KDE density: {e}")

        self.ax_kde.set_xlim(0, 1.1)
        self.ax_kde.set_xlabel("Probability Density (KDE)", color="purple")
        self.ax_kde.tick_params(axis='x', colors="purple")

    async def _read_observations_loop(self):
        while True:
            observation, status = await self._env.observe()
            if observation is None:
                self.completed = True
                self.finalize_rl_training()
                break

            self.frames_observed += 1
            
            magnitude_frame = np.abs(observation)[self._env.freq_mask].astype(np.float64)
            magnitude_frame = np.maximum(magnitude_frame, 1e-6)
            
            # --- Three-Stage Signal Denoising Pipeline ---
            # Stage 1: Spectral Median Filtering
            broadband_noise_floor = median_filter(magnitude_frame, size=self.spec_sub_kernel)
            
            denoised_magnitude = magnitude_frame - 1.2 * broadband_noise_floor
            denoised_magnitude = np.maximum(denoised_magnitude, 0.03 * broadband_noise_floor)
            denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
            
            # Stage 2: Temporal Background Subtraction
            if self.frames_observed > 20:
                temporal_background = np.mean(self.stft_magnitude_buffer[:, -20:], axis=1)
                denoised_magnitude = denoised_magnitude - self.temporal_noise_subtraction_factor * temporal_background
                denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
                
            # Stage 3: Temporal Recursive Smoothing
            if self.smoothed_magnitude is None:
                self.smoothed_magnitude = denoised_magnitude.copy()
            else:
                self.smoothed_magnitude = (self.signal_smoothing_alpha * denoised_magnitude) + \
                                          ((1.0 - self.signal_smoothing_alpha) * self.smoothed_magnitude)
            
            # Trigger NMF fitting inline
            if self.frames_observed > self.frames_per_window:
                if self._nmf_model is None:
                    await self._background_update_nmf_model()
                elif self.frames_observed % 300 == 0:
                    await self._background_update_nmf_model()

            # 2. Project activations using existing dictionary components via KL updates
            if self._nmf_model is not None:
                current_activations = self._transform_nmf_frame(self.smoothed_magnitude)
            else:
                current_activations = np.zeros((self.n_components, 1), dtype=np.float64)
                
            # 3. Slide buffers and write current frames
            self.stft_magnitude_buffer = np.roll(self.stft_magnitude_buffer, -1, axis=1)
            self.stft_magnitude_buffer[:, -1] = self.smoothed_magnitude
            
            self.activations_buffer = np.roll(self.activations_buffer, -1, axis=1)
            self.activations_buffer[:, -1] = current_activations.flatten()
            
            # Convert to Decibels for visualization
            db_frame = self._to_decibels(self.smoothed_magnitude)
            self.S_buffer = np.roll(self.S_buffer, -1, axis=1)
            self.S_buffer[:, -1] = db_frame
            
            # Advance simulation time based on hop size and sample rate
            time_per_frame = self._env.fft_hop_length / self._env.sr
            self.current_time += time_per_frame

            # --- Feature Calculation & Peak Analysis ---
            nmf_model = self._nmf_model
            if nmf_model is not None and self.frames_observed % 12 == 0:
                H = nmf_model.components_
                W = self.activations_buffer
                freqs = self._env.freqs_plot
                
                low_freq_mask = freqs <= 400
                mid_freq_mask = (freqs > 400) & (freqs <= 1000)
                high_freq_mask = (freqs > 1000) & (freqs <= 2000)
                
                for i in range(self.n_components):
                    h_i = H[i]
                    w_i = W[i]
                    
                    energy_total = np.sum(h_i)
                    if energy_total == 0: continue

                    low_freq_score = np.sum(h_i[low_freq_mask]) / energy_total
                    mid_freq_score = np.sum(h_i[mid_freq_mask]) / energy_total
                    high_freq_score = np.sum(h_i[high_freq_mask]) / energy_total
                    
                    amean = np.mean(h_i)
                    tonality_score = 1.0 - (np.exp(np.mean(np.log(h_i + 1e-9))) / amean) if amean > 0 else 0
                    
                    std_of_diffs = np.std(np.diff(w_i))
                    stability_score = 1.0 / (1.0 + std_of_diffs)

                    self.feature_scores['low_freq'][i] = low_freq_score
                    self.feature_scores['mid_freq'][i] = mid_freq_score
                    self.feature_scores['high_freq'][i] = high_freq_score
                    self.feature_scores['tonality'][i] = tonality_score
                    self.feature_scores['stability'][i] = stability_score
                    
                    freq_contribution = max(low_freq_score, mid_freq_score, high_freq_score) * 0.7
                    self.vessel_scores[i] = freq_contribution + (tonality_score * 0.2) + (stability_score * 0.1)
                    
                    self._analyze_component_peaks(i, h_i, freqs)

                if len(self.vessel_scores) > 0:
                    detections = []
                    for i in range(self.n_components):
                        score = self.vessel_scores[i]
                        raw_centroid = self.peak_centroids[i]
                        raw_spread = self.peak_spreads[i]
                        
                        if raw_centroid > 0:
                            alpha = 0.25
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
                            'spread': spread,
                            'amplitude': W[i, -1]
                        })
                    
                    self.update_multi(current_time=self.current_time, detections=detections)
            
            # Yield control back to event loop
            if self.headless:
                if self.frames_observed % 100 == 0:
                    await asyncio.sleep(0)
            else:
                await asyncio.sleep(0)

    async def _plot_update_loop(self):
        while True:
            if not plt.fignum_exists(self.fig.number):
                break
            
            status = await self._env.get_buffer_status()
            self._update_plot_data(status)
            
            self._update_annotations()
            try:
                self._update_tracker_plot()
            except Exception as e:
                print(f"Error updating tracker plot: {e}")
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            await asyncio.sleep(1/15)

    async def start(self):
        asyncio.create_task(self._read_observations_loop())
        if not self.headless:
            asyncio.create_task(self._plot_update_loop())
