import asyncio
import warnings

import librosa
import numpy as np
import matplotlib.pyplot as plt
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
        self.nmf_update_period_sec = 2.0  # Re-fit NMF model every N seconds
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
        self.fig.subplots_adjust(hspace=0.4)
        
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
        nmf_x_axis = np.linspace(-self.window_sec, 0, self.frames_per_window)
        for i in range(self.n_components):
            line, = self.ax_nmf.plot(nmf_x_axis, self.activations_buffer[i], label=f'C{i+1}')
            self.lines_nmf.append(line)
        self.ax_nmf.legend(loc='upper right')
        self.ax_nmf.set_title('NMF Activations')
        self.ax_nmf.set_xlabel("Time (s, relative to now)")
        self.ax_nmf.set_ylabel('Activation Level')
        self.ax_nmf.set_xlim(self.xaxis_extent)
        self.ax_nmf.set_ylim(0, 1)

        plt.show(block=False)

    def _to_decibels(self, magnitude_frame):
        return librosa.amplitude_to_db(magnitude_frame, ref=self._env.global_max)

    def _fit_nmf(self, data):
        """Fits the NMF model on a batch of data."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = NMF(
                n_components=self.n_components, 
                init='nndsvda',
                solver='mu',
                max_iter=self.max_iter, 
                random_state=42
            )
            # We only care about fitting the dictionary (H) here
            model.fit(data)
            return model

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

    async def _background_fit_nmf(self):
        if self._is_fitting_nmf:
            return
        
        self._is_fitting_nmf = True
        try:
            # Copy data to avoid concurrent modification issues
            data_to_fit = self.stft_magnitude_buffer.T.copy()
            
            # Fit the model in the background
            fitted_model = await asyncio.to_thread(self._fit_nmf, data_to_fit)
            
            # Update the shared model instance
            self._nmf_model = fitted_model
            
            # Re-calculate the entire activation buffer to "snap" to the new dictionary
            W = await asyncio.to_thread(self._nmf_model.transform, data_to_fit)
            self.activations_buffer[:, :] = W.T
            
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
        
        max_activation = 0
        for i, line in enumerate(self.lines_nmf):
            line.set_ydata(self.activations_buffer[i])
            max_activation = max(max_activation, np.max(self.activations_buffer[i]))
            
        current_ylim = self.ax_nmf.get_ylim()
        target_ymax = max(1e-3, max_activation * 1.1)
        if target_ymax > current_ylim[1] or target_ymax < current_ylim[1] * 0.8:
            self.ax_nmf.set_ylim(0, target_ymax)

    async def _read_observations_loop(self):
        """Continuously reads from the environment and updates data buffers."""
        while True:
            observation, status = await self._env.observe()
            if observation is None:
                await asyncio.sleep(0.01)
                continue

            self.frames_observed += 1
            
            magnitude_frame = np.abs(observation)[self._env.freq_mask]
            magnitude_frame = np.maximum(magnitude_frame, 1e-6)
            
            # Roll and update magnitude buffer
            self.stft_magnitude_buffer = np.roll(self.stft_magnitude_buffer, -1, axis=1)
            self.stft_magnitude_buffer[:, -1] = magnitude_frame
            
            # Roll and update spectrogram buffer
            frame_db = self._to_decibels(magnitude_frame)
            self.S_buffer = np.roll(self.S_buffer, -1, axis=1)
            self.S_buffer[:, -1] = frame_db
            
            # Predict the activation for the new frame and update buffer
            new_activation = await asyncio.to_thread(
                self._transform_nmf_frame, 
                magnitude_frame.reshape(-1, 1) # Needs to be 2D array
            )
            
            self.activations_buffer = np.roll(self.activations_buffer, -1, axis=1)
            self.activations_buffer[:, -1] = new_activation.flatten()

            time_per_frame = self._env.fft_hop_length / self._env.sr
            self.current_time += time_per_frame
            
            # Yield control to the event loop
            await asyncio.sleep(0)

    async def _nmf_analysis_loop(self):
        """Periodically runs the NMF analysis in the background."""
        while True:
            # Wait until the buffer is filled before the first run
            if self.frames_observed > self.frames_per_window:
                await self._background_fit_nmf()
            
            await asyncio.sleep(self.nmf_update_period_sec)

    async def _plot_update_loop(self):
        """Periodically updates the matplotlib plot."""
        while True:
            if not plt.fignum_exists(self.fig.number):
                break
            
            status = await self._env.get_buffer_status()
            self._update_plot_data(status)
            
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Yield control, allowing other tasks to run.
            await asyncio.sleep(1/15) # Aim for ~15 FPS

    async def start(self):
        """Starts all agent background tasks."""
        asyncio.create_task(self._read_observations_loop())
        asyncio.create_task(self._nmf_analysis_loop())
        asyncio.create_task(self._plot_update_loop())
