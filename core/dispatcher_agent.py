import asyncio
import warnings
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class DispatcherAgent:
    """
    The dispatcher agent definition.
    This agent observes the audio environment, performs analysis, and visualizes the results.
    """
    def __init__(self, env, min_freq, max_freq, n_fft, n_components, n_max_iter, window_sec=15.0, headless=False):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._env = env
        self.n_fft = n_fft
        self.n_components = n_components
        self.max_iter = n_max_iter
        self.headless = headless
        
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
        if self.headless:
            # Setup only the buffers and spacing constants, skipping matplotlib figures and objects
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

        plt.show(block=False)

    def _to_decibels(self, magnitude_frame):
        return librosa.amplitude_to_db(magnitude_frame, ref=self._env.global_max)

    def _transform_nmf_frame(self, frame_data):
        """Uses the NMF components to project activations for a single frame using fast numpy KL-updates."""
        if self._nmf_model is None:
            return np.zeros((self.n_components, 1), dtype=np.float64)
            
        H = self._nmf_model.components_  # shape (n_components, n_features)
        x = frame_data.flatten().astype(np.float64)  # shape (n_features,)
        
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
                    max_iter=400, 
                    random_state=42
                )
                W = await asyncio.to_thread(model.fit_transform, data_to_fit, W=W_init, H=H_init)
                H = model.components_
                
                # Normalize NMF components to max 1.0
                norms = np.max(H, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                H = H / norms
                W = W * norms.T
                
                # Sort components by spectral centroid to prevent random index permutation
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
            
            # Smoothly auto-scale the y-limit with an EMA to avoid visual jumping
            target_max = max(max_val * 1.15, 0.01)
            self.nmf_ylim_max = 0.9 * self.nmf_ylim_max + 0.1 * target_max
            self.ax_nmf.set_ylim(0, self.nmf_ylim_max)
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
            
            magnitude_frame = np.abs(observation)[self._env.freq_mask].astype(np.float64)
            magnitude_frame = np.maximum(magnitude_frame, 1e-6)
            
            # --- Three-Stage Signal Denoising Pipeline ---
            # Stage 1: Spectral Median Filtering (estimates and subtracts broadband/diffuse noise floor)
            broadband_noise_floor = median_filter(magnitude_frame, size=self.spec_sub_kernel)
            
            denoised_magnitude = magnitude_frame - 1.2 * broadband_noise_floor
            denoised_magnitude = np.maximum(denoised_magnitude, 0.03 * broadband_noise_floor)
            denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
            
            # Stage 2: Temporal Background Subtraction (removes stationary equipment/grid hums)
            if self.frames_observed > 20:
                temporal_background = np.mean(self.stft_magnitude_buffer[:, -20:], axis=1)
                denoised_magnitude = denoised_magnitude - self.temporal_noise_subtraction_factor * temporal_background
                denoised_magnitude = np.maximum(denoised_magnitude, 1e-6)
                
            # Stage 3: Temporal Recursive Smoothing (reduces frame-to-frame NMF activation jitter)
            if self.smoothed_magnitude is None:
                self.smoothed_magnitude = denoised_magnitude.copy()
            else:
                self.smoothed_magnitude = (self.signal_smoothing_alpha * denoised_magnitude) + \
                                          ((1.0 - self.signal_smoothing_alpha) * self.smoothed_magnitude)
            
            # Trigger NMF fitting inline
            if self.frames_observed > self.frames_per_window:
                if self._nmf_model is None:
                    # Fit immediately for the first time
                    await self._background_update_nmf_model()
                elif self.frames_observed % 50 == 0:
                    # Trigger a background update every 50 frames
                    asyncio.create_task(self._background_update_nmf_model())

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
            
            # Yield control back to event loop
            await asyncio.sleep(0)



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
        if not self.headless:
            asyncio.create_task(self._plot_update_loop())
