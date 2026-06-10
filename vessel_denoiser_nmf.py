import os
import glob
import asyncio
import collections
import warnings
import time
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

# --- Configuration Constants ---
# Override by setting the RECORDINGS_DIR environment variable
BASE_DIR = os.environ.get("RECORDINGS_DIR", "D:/RoyStudies/Recordings")
croatia_base_dir = f"{BASE_DIR}/Croatia/Ocean Sonics"
FILE_DIR = f"{croatia_base_dir}/2507_1"

MIN_FREQ = 10  # High-pass threshold to eliminate ocean swell noise
MAX_FREQ = 2000  # Focus on low-mid frequency vessel harmonics
N_FFT = 32 * 1024  # Lower FFT size for rapid computation
N_COMPONENTS = 4  # Expected maximum number of independent sound sources
WINDOW_SEC = 15.0  # Moving visual window width in seconds
N_MAX_ITER = 400

# Extreme Playback Speed Control (1.0 = Real-time, 2000.0 = Blazing speed)
PLAYBACK_SPEED = 5.0

# Ensure directory exists for running sample
if not os.path.exists(FILE_DIR):
    os.makedirs(FILE_DIR)
    print(f"Created '{FILE_DIR}' directory. Please drop your .wav files there.")


class AudioEnvironment:
    """Streams sequential audio files as continuous frame-by-frame STFT data."""

    def __init__(self, file_dir, min_freq, max_freq, n_fft, max_buffered_sec=120):
        self.file_dir = file_dir
        self.wav_files = sorted(glob.glob(os.path.join(file_dir, "*.wav")))
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_fft = n_fft
        self.fft_hop_length = n_fft // 4

        if not self.wav_files:
            raise ValueError(f"No .wav files found in directory: {self.file_dir}")

        self.info = sf.info(self.wav_files[0])
        self.sr = self.info.samplerate

        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.freq_mask = (self.freqs >= self.min_freq) & (self.freqs <= self.max_freq)
        self.freqs_plot = self.freqs[self.freq_mask]

        self.chunk_buffer = collections.deque()
        self.max_buffered_chunks = max_buffered_sec * librosa.time_to_frames(
            1.0, sr=self.sr, hop_length=self.fft_hop_length
        )
        self.file_index = 0
        self.reading = True
        self.buffer_lock = asyncio.Lock()
        self.data_available = asyncio.Event()
        self.global_max = 1e-6

    async def _read_files_loop_fft(self):
        """Streams audio files and pre-computes their STFT instantly in vectorized C-code."""
        while self.file_index < len(self.wav_files):
            async with self.buffer_lock:
                buffer_size = len(self.chunk_buffer)

            # Throttle file reader if the buffer exceeds safety capacity limits
            if buffer_size >= self.max_buffered_chunks:
                await asyncio.sleep(0.1)
                continue

            file_path = self.wav_files[self.file_index]
            print(f"[Environment] Vectorized STFT computation for: {os.path.basename(file_path)}")

            try:
                def compute_file_stft():
                    # Load whole audio file at once (extremely fast in soundfile C-code)
                    with sf.SoundFile(file_path) as f:
                        data = f.read(dtype='float32')
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1)

                    # Compute entire STFT matrix in a single optimized vectorized operation
                    fft_matrix = librosa.stft(
                        data,
                        n_fft=self.n_fft,
                        hop_length=self.fft_hop_length,
                        center=True
                    )
                    # Convert to a list of spectral columns (frames)
                    return list(fft_matrix.T)

                # Offload STFT matrix calculation to background thread to prevent UI freezing
                file_frames = await asyncio.to_thread(compute_file_stft)

                async with self.buffer_lock:
                    self.chunk_buffer.extend(file_frames)
                    self.data_available.set()

            except Exception as e:
                print(f"Error streaming {file_path}: {e}")

            self.file_index += 1

        self.reading = False
        self.data_available.set()

    async def start(self):
        asyncio.create_task(self._read_files_loop_fft())

    async def get_buffer_status(self):
        async with self.buffer_lock:
            size = len(self.chunk_buffer)
            percentage = (size / self.max_buffered_chunks) * 100 if self.max_buffered_chunks > 0 else 0
            return size, percentage

    async def observe_batch(self, max_batch_size=512):
        """Pops a batch of frames from the buffer in a single locked transaction."""
        frames = []
        async with self.buffer_lock:
            while self.chunk_buffer and len(frames) < max_batch_size:
                frames.append(self.chunk_buffer.popleft())

        if frames:
            # Vectorized local max updating for normalization dB scaling
            local_maxs = [np.max(np.abs(f)) for f in frames]
            local_max = np.max(local_maxs)
            self.global_max = max(0.95 * self.global_max, local_max)
            if self.global_max == 0:
                self.global_max = 1.0

        status = await self.get_buffer_status()
        return frames, status


class DenoisingDispatcherAgent:
    """Denoises raw audio streaming frames, updates a moving plot, and tracks source separation."""

    def __init__(self, env, min_freq, max_freq, n_fft, n_components, max_iter, window_sec=15.0):
        self._env = env
        self.n_components = n_components
        self.max_iter = max_iter
        self.window_sec = window_sec
        self.current_time = 0.0
        self.frames_observed = 0

        self._is_fitting_nmf = False
        self.nmf_update_period_sec = 4.0
        self._nmf_model = None

        # Playback speed controller (dynamically pacing visual updates)
        self.playback_speed = PLAYBACK_SPEED

        # Temporal smoothing factor for the denoised magnitude (smoothes out frame-to-frame jumping)
        self.signal_smoothing_alpha = 0.12
        self.smoothed_magnitude = None

        # Determine the adaptive median filter kernel size for spectral background subtraction
        num_bins = len(self._env.freqs_plot)
        self.spec_sub_kernel = min(51, num_bins // 3)
        if self.spec_sub_kernel % 2 == 0:
            self.spec_sub_kernel += 1
        if self.spec_sub_kernel < 3:
            self.spec_sub_kernel = 3

        self._setup_plot()

    def _setup_plot(self):
        plt.ion()
        self.fig, (self.ax_spec, self.ax_nmf) = plt.subplots(
            2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 2]}
        )
        self.fig.subplots_adjust(hspace=0.4)

        # Determine sampling allocation sizing
        frames_per_sec_display = librosa.time_to_frames(
            1.0, sr=self._env.sr, hop_length=self._env.fft_hop_length
        )
        self.frames_per_window = int(self.window_sec * frames_per_sec_display)
        num_freq_bins = len(self._env.freqs_plot)

        # Buffers holding historical data inside the sliding time window
        self.S_buffer = np.full((num_freq_bins, self.frames_per_window), -80.0)
        self.stft_magnitude_buffer = np.full((num_freq_bins, self.frames_per_window), 1e-6, dtype=np.float64)
        self.activations_buffer = np.zeros((self.n_components, self.frames_per_window), dtype=np.float64)

        # Change extent to start at absolute zero up to window_sec length
        extent = [0, self.window_sec, self._env.freqs_plot[0], self._env.freqs_plot[-1]]

        self.img = self.ax_spec.imshow(
            self.S_buffer, aspect='auto', origin='lower', cmap='magma', extent=extent,
            vmin=-70, vmax=-40, interpolation='bilinear'
        )

        # Initialize limits at absolute starting frame boundaries
        self.ax_spec.set_xlim(0, self.window_sec)
        self.title = self.ax_spec.set_title("Live Moving Spectrogram")
        self.ax_spec.set_xlabel("Time (Seconds)")
        self.ax_spec.set_ylabel("Frequency (Hz)")
        self.fig.colorbar(self.img, ax=self.ax_spec, format="%+2.0f dB")

        # NMF Plot Lines Configuration
        self.lines_nmf = []
        nmf_x_axis = self._env.freqs_plot
        for i in range(self.n_components):
            line, = self.ax_nmf.plot(nmf_x_axis, np.zeros(num_freq_bins), label=f'Vessel Class {i + 1}')
            self.lines_nmf.append(line)
        self.ax_nmf.legend(loc='upper right')
        self.ax_nmf.set_title('Extracted Harmonic Profiles')
        self.ax_nmf.set_xlabel("Frequency (Hz)")
        self.ax_nmf.set_ylabel('Normalized Scale Factor')
        self.ax_nmf.set_xlim(self._env.freqs_plot[0], self._env.freqs_plot[-1])
        self.nmf_ylim_max = 0.15
        self.ax_nmf.set_ylim(0, self.nmf_ylim_max)

        plt.show(block=False)

    def _to_decibels(self, magnitude_frame):
        return librosa.amplitude_to_db(magnitude_frame, ref=self._env.global_max)

    async def _background_update_nmf_model(self):
        """Updates NMF Dictionary matrices asynchronously based entirely on denoised buffers."""
        if self._is_fitting_nmf:
            return
        self._is_fitting_nmf = True
        try:
            data_to_fit = self.stft_magnitude_buffer.T.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)

                freqs = self._env.freqs_plot
                num_freqs = len(freqs)

                # Initialization tracking to guarantee lower convergence profiles
                H_init = np.random.rand(self.n_components, num_freqs) * 0.01 + 0.01
                centers_norm = np.power(np.linspace(0, 1, self.n_components), 2)
                centers = centers_norm * (freqs[-1] - freqs[0]) + freqs[0]
                sigma = (freqs[-1] - freqs[0]) / (self.n_components * 3)

                for i in range(self.n_components):
                    bump = np.exp(-0.5 * ((freqs - centers[i]) / sigma) ** 2)
                    H_init[i] += bump

                W_init = np.random.rand(data_to_fit.shape[0], self.n_components) * 0.1 + 0.1

                model = NMF(
                    n_components=self.n_components, init='custom', solver='mu',
                    beta_loss='kullback-leibler', max_iter=self.max_iter, random_state=42
                )
                W = await asyncio.to_thread(model.fit_transform, data_to_fit, W=W_init, H=H_init)
                H = model.components_

                norms = np.max(H, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                H = H / norms
                W = W * norms.T

                # Sorting components by spectral centroid to resolve structural permutation issues
                centroids = np.sum(H * freqs, axis=1) / (np.sum(H, axis=1) + 1e-9)
                sort_indices = np.argsort(centroids)

                model.components_ = H[sort_indices]
                W_sorted = W[:, sort_indices]

                self._nmf_model = model
                self.activations_buffer[:, :] = W_sorted.T
        except Exception as e:
            print(f"NMF Exception Encountered: {e}")
        finally:
            self._is_fitting_nmf = False

    def _update_plot_data(self, status):
        # Update raw matrix content
        self.img.set_data(self.S_buffer)

        # Dynamically calculate the sliding extent bounds based on current runtime
        t_start = max(0.0, self.current_time - self.window_sec)
        t_end = max(self.window_sec, self.current_time)

        # Redraw image bounds coordinates gradually without replacing data blocks
        self.img.set_extent([t_start, t_end, self._env.freqs_plot[0], self._env.freqs_plot[-1]])
        self.ax_spec.set_xlim(t_start, t_end)

        buffer_size, buffer_percentage = status
        self.title.set_text(
            f"Live Gradual Waterfall (Speed: {self.playback_speed}x) | Absolute Time: {self.current_time:.2f}s | "
            f"Disk Cache Buffer: {buffer_percentage:.0f}% ({buffer_size} frames)"
        )

        # Smoothly track lines matching current activations
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
        """Processes continuous chunks in highly optimized batches to handle extreme playback speeds."""
        start_real_time = time.perf_counter()
        start_sim_time = self.current_time
        time_per_frame = self._env.fft_hop_length / self._env.sr

        while True:
            # Determine target timeline position in simulation space
            elapsed_real = time.perf_counter() - start_real_time
            target_sim_time = start_sim_time + (elapsed_real * self.playback_speed)

            # Number of frames we need to catch up
            needed_frames = int((target_sim_time - self.current_time) / time_per_frame)

            if needed_frames <= 0:
                await asyncio.sleep(0.005)
                continue

            # Dynamically size the processing batch (max 512 frames per loop)
            batch_size = min(needed_frames, 512)

            # Pop frames in a single batched operation
            frames, status = await self._env.observe_batch(batch_size)

            if not frames:
                if not self._env.reading:
                    # File streaming is complete
                    break
                # Wait for data to arrive from the reader thread
                await self._env.data_available.wait()
                # Re-sync timeline reference to prevent massive burst jumps after a loading delay
                start_real_time = time.perf_counter()
                start_sim_time = self.current_time
                continue

            batch_len = len(frames)

            # Convert the batch into a 2D matrix shape (num_freqs, batch_len)
            raw_magnitudes = np.abs(np.array(frames)).T[self._env.freq_mask].astype(np.float64)

            # --- Vectorized Spectral Median Filtering (Two-Pass Background Estimation) ---
            # Instead of a temporal average (which eats stable ship tones), we estimate the
            # broadband noise floor across the frequency axis for each frame. This preserves
            # all sharp narrowband vessel harmonics perfectly while stripping ambient background.
            broadband_noise_floor = median_filter(raw_magnitudes, size=(self.spec_sub_kernel, 1))

            # Subtract the estimated broadband background noise floor
            # Using 1.2x over-subtraction to completely clean up ambient ocean noise
            denoised_magnitudes = raw_magnitudes - 1.2 * broadband_noise_floor
            denoised_magnitudes = np.maximum(denoised_magnitudes, 0.03 * broadband_noise_floor)
            denoised_magnitudes = np.maximum(denoised_magnitudes, 1e-6)

            smoothed_magnitudes = np.zeros_like(raw_magnitudes)

            # Denoise the columns using our temporal recursive smoothing filter (makes visual transitions buttery)
            for i in range(batch_len):
                denoised_col = denoised_magnitudes[:, i]
                if self.smoothed_magnitude is None:
                    self.smoothed_magnitude = denoised_col.copy()
                else:
                    self.smoothed_magnitude = (self.signal_smoothing_alpha * denoised_col) + \
                                              ((1.0 - self.signal_smoothing_alpha) * self.smoothed_magnitude)
                smoothed_magnitudes[:, i] = self.smoothed_magnitude

            # Roll and write to spectrogram buffer in a single batched matrix write
            self.stft_magnitude_buffer = np.roll(self.stft_magnitude_buffer, -batch_len, axis=1)
            self.stft_magnitude_buffer[:, -batch_len:] = smoothed_magnitudes

            # Vectorized decibel conversion of the whole batch
            frames_db = self._to_decibels(smoothed_magnitudes)
            self.S_buffer = np.roll(self.S_buffer, -batch_len, axis=1)
            self.S_buffer[:, -batch_len:] = frames_db

            # Batched NMF Projection (Converts whole batch in a single background operation)
            if self._nmf_model is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        # transform expects: (num_samples, num_features) -> (batch_len, num_freqs)
                        new_activations = await asyncio.to_thread(
                            self._nmf_model.transform, smoothed_magnitudes.T.astype(np.float64)
                        )
                        new_activations_t = new_activations.T  # (num_components, batch_len)
                except Exception as e:
                    print(f"NMF Transform failed: {e}")
                    new_activations_t = np.zeros((self.n_components, batch_len), dtype=np.float64)
            else:
                new_activations_t = np.zeros((self.n_components, batch_len), dtype=np.float64)

            self.activations_buffer = np.roll(self.activations_buffer, -batch_len, axis=1)
            self.activations_buffer[:, -batch_len:] = new_activations_t

            self.frames_observed += batch_len
            self.current_time += batch_len * time_per_frame

            # Let other tasks run for a millisecond
            await asyncio.sleep(0.001)

    async def _nmf_analysis_loop(self):
        while True:
            if self.frames_observed > self.frames_per_window:
                await self._background_update_nmf_model()
            await asyncio.sleep(self.nmf_update_period_sec)

    async def _plot_update_loop(self):
        """Updates the physical screen output at a steady, buttery-smooth refresh rate."""
        while True:
            if not plt.fignum_exists(self.fig.number):
                print("[Application] Visualization window closed by user.")
                break

            status = await self._env.get_buffer_status()
            self._update_plot_data(status)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            await asyncio.sleep(1 / 30)  # Steady 30 Hz refresh rate

    async def start(self):
        asyncio.create_task(self._read_observations_loop())
        asyncio.create_task(self._nmf_analysis_loop())
        await self._plot_update_loop()


async def main():
    print("=== Launching Underwater Acoustic Denoising Analysis Pipeline ===")

    try:
        env = AudioEnvironment(
            file_dir=FILE_DIR,
            min_freq=MIN_FREQ,
            max_freq=MAX_FREQ,
            n_fft=N_FFT
        )
    except ValueError as e:
        print(f"\n[Error] {e}")
        print(f"Please copy your hydrophone recorded .wav tracks directly inside: '{os.path.abspath(FILE_DIR)}'")
        return

    agent = DenoisingDispatcherAgent(
        env=env,
        min_freq=MIN_FREQ,
        max_freq=MAX_FREQ,
        n_fft=N_FFT,
        n_components=N_COMPONENTS,
        max_iter=N_MAX_ITER,
        window_sec=WINDOW_SEC
    )

    # Boot environment file processing loops
    await env.start()
    # Handle rendering and internal matrix factorization tracking updates
    await agent.start()


if __name__ == "__main__":
    # Ensure correct event loops are handled across async implementations cleanly
    asyncio.run(main())