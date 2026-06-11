import asyncio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from core.vessel_state import VesselState, VesselStateTracker

class SignalProcessorAgent:
    def __init__(self, dispatcher_agent, proximity_threshold_hz=65.0, association_threshold_hz=80.0, peak_spread_window_bins=15, variance_multiplier=1.5, min_variance_floor=25.0, consolidation_threshold_hz=65.0):
        self._dispatcher = dispatcher_agent
        self.n_components = self._dispatcher.n_components
        self._analysis_interval_sec = 1.0
        self.peak_spread_window_bins = peak_spread_window_bins
        self.variance_multiplier = variance_multiplier
        self.min_variance_floor = min_variance_floor
        
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
        
        self._annotations = []
        if not self._dispatcher.headless:
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
        self.tracker = VesselStateTracker(
            drift_threshold_hz=15.0, 
            min_vessel_score=0.45,
            proximity_threshold_hz=proximity_threshold_hz,
            association_threshold_hz=association_threshold_hz,
            consolidation_threshold_hz=consolidation_threshold_hz
        )
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
            # Always look at the primary (lowest-frequency) peak first
            dom_peak = peaks[0]
            
            # Measure spread strictly in a local window
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

    async def _analysis_loop(self):
        """Periodically calculates Vessel Score and peak stats for each component."""
        last_analysis_time = 0.0
        while not self._dispatcher.completed:
            curr_time = self._dispatcher.current_time
            if curr_time - last_analysis_time >= self._analysis_interval_sec:
                nmf_model = self._dispatcher._nmf_model
                if nmf_model is not None:
                    H = nmf_model.components_
                    W = self._dispatcher.activations_buffer
                    freqs = self._dispatcher._env.freqs_plot
                    
                    low_freq_mask = freqs <= 400
                    mid_freq_mask = (freqs > 400) & (freqs <= 1000)
                    high_freq_mask = (freqs > 1000) & (freqs <= 2000)
                    
                    for i in range(self.n_components):
                        h_i = H[i]
                        w_i = W[i]
                        
                        # --- Feature Calculation ---
                        energy_total = np.sum(h_i)
                        if energy_total == 0: continue

                        low_freq_score = np.sum(h_i[low_freq_mask]) / energy_total
                        mid_freq_score = np.sum(h_i[mid_freq_mask]) / energy_total
                        high_freq_score = np.sum(h_i[high_freq_mask]) / energy_total
                        
                        amean = np.mean(h_i)
                        tonality_score = 1.0 - (np.exp(np.mean(np.log(h_i + 1e-9))) / amean) if amean > 0 else 0
                        
                        std_of_diffs = np.std(np.diff(w_i))
                        stability_score = 1.0 / (1.0 + std_of_diffs)

                        # --- Store Feature Scores ---
                        self.feature_scores['low_freq'][i] = low_freq_score
                        self.feature_scores['mid_freq'][i] = mid_freq_score
                        self.feature_scores['high_freq'][i] = high_freq_score
                        self.feature_scores['tonality'][i] = tonality_score
                        self.feature_scores['stability'][i] = stability_score
                        
                        # --- Final Weighted Vessel Score ---
                        freq_contribution = max(low_freq_score, mid_freq_score, high_freq_score) * 0.7
                        self.vessel_scores[i] = freq_contribution + (tonality_score * 0.2) + (stability_score * 0.1)
                        
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
                                'spread': spread,
                                'amplitude': W[i, -1]
                            })
                        
                        self.tracker.update_multi(
                            current_time=curr_time,
                            detections=detections
                        )
                last_analysis_time = curr_time
            await asyncio.sleep(0.01)
            
    def _update_annotations(self):
        """Updates plot annotations with scores and peak stats."""
        if self._dispatcher.headless:
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
            
            self._dispatcher.lines_nmf[i].set_color(plt.cm.coolwarm(score))

    def _update_tracker_plot(self):
        """Draws the speed timeline, the probability density curve, and the signal amplitude profile."""
        if not hasattr(self._dispatcher, 'ax_track') or not plt.fignum_exists(self._dispatcher.fig.number):
            return
            
        # 1. Clear axes
        self._dispatcher.ax_track.clear()
        self._dispatcher.ax_kde.clear()
        if hasattr(self._dispatcher, 'ax_amp'):
            self._dispatcher.ax_amp.clear()

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

        # 5. Sort Vessel IDs by cumulative duration (most active first)
        vessel_durations.sort(key=lambda x: x[1], reverse=True)

        labeled_vessels = set()
        vessel_kde_data = []

        for rank, (vessel_id, total_dur, states) in enumerate(vessel_durations):
            is_dominant = (rank < 8)  # top 8 are dominant
            
            # Select color based on rank
            color = plt.cm.tab10(rank % 10)
            
            meta = {
                'vessel_id': vessel_id,
                'is_dominant': is_dominant,
                'color': color,
                'label': f"{vessel_id} ({total_dur:.1f}s)" if is_dominant else "Noise"
            }

            sorted_segs = sorted(states, key=lambda s: s.start_time)
            
            # Plot each individual segment
            for state in sorted_segs:
                is_active = (state.end_time is None)
                start_t = state.start_time
                end_t = state.end_time if state.end_time else self._dispatcher.current_time
                duration = end_t - start_t
                
                mean_f = state.mean_frequency
                scaled_variance = max(state.total_variance * self.variance_multiplier, self.min_variance_floor)
                std_f = np.sqrt(scaled_variance)
                
                t_seg = np.linspace(start_t, end_t, len(state.amplitudes)) if len(state.amplitudes) > 1 else np.array([start_t])
                
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
                    
                    # Plot amplitude line on ax_amp
                    if hasattr(self._dispatcher, 'ax_amp'):
                        self._dispatcher.ax_amp.plot(
                            t_seg, state.amplitudes,
                            color=color, linestyle=linestyle, linewidth=2.0, alpha=alpha_val
                        )
                        
                    # Save frequency data points for KDE density estimation
                    for freq_val in state.frequencies:
                        vessel_kde_data.append((freq_val, color))
                else:
                    # Minor/transient noise: draw as a thin, faded grey background line
                    self._dispatcher.ax_track.plot(
                        [start_t, end_t], [mean_f, mean_f], 
                        color=color, linestyle="-", linewidth=1.0, alpha=0.25
                    )
                    # Plot minor amplitude line on ax_amp
                    if hasattr(self._dispatcher, 'ax_amp'):
                        self._dispatcher.ax_amp.plot(
                            t_seg, state.amplitudes,
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

        # 6. Draw vertical line indicating current tracking timeline position
        self._dispatcher.ax_track.axvline(self._dispatcher.current_time, color="red", linestyle="--", linewidth=1.0)
        if hasattr(self._dispatcher, 'ax_amp'):
            self._dispatcher.ax_amp.axvline(self._dispatcher.current_time, color="red", linestyle="--", linewidth=1.0)

        # Update absolute x-limits dynamically
        t_limit = max(15.0, self._dispatcher.current_time + 1.0)
        self._dispatcher.ax_track.set_xlim(0, t_limit)
        if hasattr(self._dispatcher, 'ax_amp'):
            self._dispatcher.ax_amp.set_xlim(0, t_limit)

        # Plot RL decisions (detections vs. misdirections) if history exists
        if hasattr(self.tracker, 'rl_history') and self.tracker.rl_history:
            times_ok, freqs_ok = [], []
            times_fn, freqs_fn = [], []
            times_dup, freqs_dup = [], []
            times_bad, freqs_bad = [], []
            
            for pt in self.tracker.rl_history:
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
            
            if times_ok:
                self._dispatcher.ax_track.scatter(times_ok, freqs_ok, color="green", s=25, alpha=0.7, marker="o", label="Correct Detection")
            if times_fn:
                self._dispatcher.ax_track.scatter(times_fn, freqs_fn, color="red", s=35, alpha=0.9, marker="x", label="False Negative (Miss)")
            if times_dup:
                self._dispatcher.ax_track.scatter(times_dup, freqs_dup, color="orange", s=35, alpha=0.9, marker="^", label="Duplicate Spawn")
            if times_bad:
                self._dispatcher.ax_track.scatter(times_bad, freqs_bad, color="magenta", s=35, alpha=0.9, marker="s", label="Bad Association")

        # Show legend
        self._dispatcher.ax_track.legend(loc="upper left", fontsize=7, framealpha=0.6)

        # 7. Render Probability Density curve (KDE) on twin y-axis
        if len(vessel_kde_data) > 3:
            try:
                from scipy.stats import gaussian_kde
                freq_vals = [pt[0] for pt in vessel_kde_data]
                kde = gaussian_kde(freq_vals, bw_method=0.1)
                
                # Compute density curve
                y_grid = np.linspace(self._dispatcher.min_freq, self._dispatcher.max_freq, 500)
                density = kde(y_grid)
                
                # Normalize density curve to maximum width of 0.8
                max_d = np.max(density)
                if max_d > 0:
                    density = (density / max_d) * 0.8
                    
                # Plot density curve
                self._dispatcher.ax_kde.plot(density, y_grid, color="purple", linewidth=1.5, alpha=0.7)
                self._dispatcher.ax_kde.fill_betweenx(y_grid, 0, density, color="purple", alpha=0.1)
            except Exception as e:
                print(f"Error drawing KDE density: {e}")

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
        if not self._dispatcher.headless:
            asyncio.create_task(self._plot_update_loop())
