import asyncio
import warnings

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


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

        # Visualization setup
        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        
        self.fig, (self.ax_spec, self.ax_nmf) = plt.subplots(
            2, 1, 
            figsize=(12, 8),
            gridspec_kw={'height_ratios': [3, 2]}
        )
        self.fig.subplots_adjust(hspace=0.5)
        
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
                await asyncio.sleep(0.01)
                continue

            self.frames_observed += 1
            
            # Make sure it's float64 before assigning to buffer
            magnitude_frame = np.abs(observation)[self._env.freq_mask].astype(np.float64)
            magnitude_frame = np.maximum(magnitude_frame, 1e-6)
            
            self.stft_magnitude_buffer = np.roll(self.stft_magnitude_buffer, -1, axis=1)
            self.stft_magnitude_buffer[:, -1] = magnitude_frame
            
            frame_db = self._to_decibels(magnitude_frame)
            self.S_buffer = np.roll(self.S_buffer, -1, axis=1)
            self.S_buffer[:, -1] = frame_db
            
            new_activation = await asyncio.to_thread(
                self._transform_nmf_frame, 
                magnitude_frame.reshape(-1, 1)
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

    def _analyze_component_peaks(self, component_index, H_component, freqs):
        """Finds peaks in an NMF component and calculates their weighted statistics."""
        max_h = np.max(H_component)
        if max_h == 0:
            self.peak_centroids[component_index] = 0
            self.peak_spreads[component_index] = 0
            return

        # Find significant peaks
        peaks, _ = find_peaks(H_component, height=max_h * 0.05, prominence=max_h * 0.02)
        
        n_peaks = len(peaks)
        if n_peaks > 0:
            peak_freqs = freqs[peaks]
            peak_weights = H_component[peaks]
            
            # Use peak heights as weights for the mean and variance
            weight_sum = np.sum(peak_weights)
            if weight_sum > 0:
                centroid = np.sum(peak_freqs * peak_weights) / weight_sum
                variance = np.sum((peak_freqs - centroid)**2 * peak_weights) / weight_sum
                spread = np.sqrt(variance)
                
                self.peak_centroids[component_index] = centroid
                self.peak_spreads[component_index] = spread
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

    async def _plot_update_loop(self):
        while True:
            if not self._dispatcher.fig.canvas.manager.window:
                break
            
            self._update_annotations()
            await asyncio.sleep(self._analysis_interval_sec)

    async def start(self):
        asyncio.create_task(self._analysis_loop())
        asyncio.create_task(self._plot_update_loop())
