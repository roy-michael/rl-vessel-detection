import os
import glob
import asyncio
import collections

import librosa
import numpy as np
import soundfile as sf


class Environment:
    _state = None
    _agents = []

    def __init__(self,
                 file_dir,
                 min_freq=0,
                 max_freq=2000,
                 n_fft=32 * 1024,
                 hop_length=1024,
                 max_buffered_sec=120,
                 max_files=None):

        self.file_dir = file_dir
        self.wav_files = sorted(glob.glob(os.path.join(file_dir, "*.wav")))
        if max_files is not None:
            self.wav_files = self.wav_files[:max_files]
            
        # Extract base real timestamp from the first WAV file
        self.start_timestamp = 0.0
        if self.wav_files:
            import re
            from datetime import datetime
            first_file = os.path.basename(self.wav_files[0])
            match = re.search(r"(\d{8})_(\d{6})", first_file)
            if match:
                try:
                    dt = datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")
                    self.start_timestamp = dt.timestamp()
                except Exception as e:
                    print(f"Error parsing start time from {first_file}: {e}")
                    
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fft_hop_length = hop_length

        if not self.wav_files:
            raise ValueError(f"No .wav files found in {self.file_dir}")

        self.info = sf.info(self.wav_files[0])
        self.sr = self.info.samplerate
        self.samples_per_second = int(self.sr * 1.0)

        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.freq_mask = (self.freqs >= self.min_freq) & (self.freqs <= self.max_freq)
        self.freqs_plot = self.freqs[self.freq_mask]

        # Buffer will now store single STFT frames (columns)
        self.chunk_buffer = collections.deque()
        self.max_buffered_chunks = max_buffered_sec * librosa.time_to_frames(1.0, sr=self.sr,
                                                                             hop_length=self.fft_hop_length)
        self.file_index = 0
        self.reading = True

        self.buffer_lock = asyncio.Lock()
        self.data_available = asyncio.Event()

        self.global_max = 1e-6

    async def _read_files_loop_fft(self):
        cache_dir = os.path.join(self.file_dir, ".stft_cache")
        os.makedirs(cache_dir, exist_ok=True)

        while self.file_index < len(self.wav_files):
            async with self.buffer_lock:
                buffer_size = len(self.chunk_buffer)

            # If the buffer is full, wait before loading more files
            if buffer_size >= self.max_buffered_chunks:
                await asyncio.sleep(0.1)
                continue

            file_path = self.wav_files[self.file_index]
            base_name = os.path.basename(file_path)
            cache_file = os.path.join(cache_dir, f"{base_name}_{self.n_fft}_{self.fft_hop_length}.npy")

            try:
                if os.path.exists(cache_file):
                    fft = await asyncio.to_thread(np.load, cache_file)
                else:
                    # Load the entire audio file into memory using fast soundfile read
                    data, _ = await asyncio.to_thread(sf.read, file_path, dtype='float32')
                    if len(data.shape) > 1:
                        y = np.mean(data, axis=1)
                    else:
                        y = data
                    # Perform STFT on the whole file
                    fft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.fft_hop_length)
                    
                    # Cache it safely
                    tmp_file = cache_file.replace(".npy", "_tmp.npy")
                    await asyncio.to_thread(np.save, tmp_file, fft)
                    os.replace(tmp_file, cache_file)

                async with self.buffer_lock:
                    # Extend the buffer with all frames (columns) from the STFT, transposed
                    self.chunk_buffer.extend(fft.T)
                    self.data_available.set()

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

            self.file_index += 1

        self.reading = False
        self.data_available.set()

    async def start(self):
        # Start background reading task
        asyncio.create_task(self._read_files_loop_fft())

    def reset(self):
        self.file_index = 0
        self.reading = True
        self.chunk_buffer.clear()
        self.data_available.clear()

    async def get_buffer_status(self):
        async with self.buffer_lock:
            size = len(self.chunk_buffer)
            percentage = (size / self.max_buffered_chunks) * 100 if self.max_buffered_chunks > 0 else 0
            return size, percentage

    async def _get_frame(self, remove_from_buffer=True):
        frame = None
        # Wait until a frame is available or reading is finished
        while frame is None:
            async with self.buffer_lock:
                if self.chunk_buffer:
                    if remove_from_buffer:
                        frame = self.chunk_buffer.popleft()
                    else:
                        frame = self.chunk_buffer[0]  # Peek at the first item
                elif not self.reading:
                    return None, None  # No more data and finished reading
                else:
                    self.data_available.clear()

            if frame is None and self.reading:
                await self.data_available.wait()

        if frame is not None:
            # Update global max for dB scaling
            local_max = np.max(np.abs(frame))
            self.global_max = max(0.95 * self.global_max, local_max)
            if self.global_max == 0:
                self.global_max = 1.0

        return frame, await self.get_buffer_status()

    async def observe(self):
        """Gets the next STFT frame and removes it from the buffer."""
        return await self._get_frame(remove_from_buffer=True)

    async def peek(self):
        """Gets the next STFT frame without removing it from the buffer."""
        return await self._get_frame(remove_from_buffer=False)

    def act(self):
        pass
