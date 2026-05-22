import os
import glob
import psutil
import asyncio

import librosa
import matplotlib.pyplot as plt
import numpy as np

from agent import DispatcherAgent
from environment import Environment


def translate(obsrv, env):
    """
    Translates a raw STFT slice into a spectrogram slice for plotting.
    The STFT is now done in the environment.
    """
    # Convert to decibels.
    S_db = librosa.amplitude_to_db(np.abs(obsrv), ref=env.global_max)

    # Apply frequency mask.
    S_db_plot = S_db[env.freq_mask]
    return S_db_plot


async def main():
    base_dir = "D:/RoyStudies/Recordings"
    croatia_base_dir = f"{base_dir}/Croatia/Ocean Sonics"
    file_dir = f"{croatia_base_dir}/2507_1"
    min_freq = 0
    max_freq = 2000
    n_fft = 32 * 1024
    hop_length = int(4 * 1024)

    try:
        env = Environment(file_dir, min_freq, max_freq, n_fft, hop_length)
        agent = DispatcherAgent(env, min_freq, max_freq, n_fft, 10, 2000)

    except ValueError as e:
        print(e)
        return

    await env.start()
    await agent.start()

    window_sec = 15.0
    
    # Enable interactive mode for a moving timeline
    plt.ion()
    
    # Create subplots for the spectrogram and the system metrics
    fig, (ax_spec, ax_metrics) = plt.subplots(
        2, 1, 
        figsize=(12, 8),
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.subplots_adjust(hspace=0.4)
    
    # Setup Metrics Plot
    metrics_buffer_size = 100
    metrics_time = np.linspace(-window_sec, 0, metrics_buffer_size)
    cpu_buffer = np.zeros(metrics_buffer_size)
    disk_buffer = np.zeros(metrics_buffer_size)
    process_buffer = np.zeros(metrics_buffer_size)
    
    line_cpu, = ax_metrics.plot(metrics_time, cpu_buffer, label="CPU (%)", color="red", animated=True)
    line_disk, = ax_metrics.plot(metrics_time, disk_buffer, label="Disk I/O (%)", color="blue", animated=True)
    
    ax_metrics2 = ax_metrics.twinx()
    line_proc, = ax_metrics2.plot(metrics_time, process_buffer, label="Processes", color="green", animated=True)
    
    xaxis_extent = [-window_sec, 0]
    ax_metrics.set_xlim(xaxis_extent)
    ax_metrics.set_ylim(0, 100)
    ax_metrics.set_xlabel("Time (s, relative to now)")
    ax_metrics.set_ylabel("Load (%)")
    ax_metrics2.set_ylabel("Process Count")
    ax_metrics.set_title("System Performance Metrics")
    
    # Combine legends
    lines = [line_cpu, line_disk, line_proc]
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="upper left")

    disk_io_start = psutil.disk_io_counters()

    current_time = 0.0
    metrics_update_counter = 0

    # Determine the number of frames needed for the display window
    frames_per_sec_display = librosa.time_to_frames(1.0, sr=env.sr, hop_length=env.hop_length)
    frames_per_window = int(window_sec * frames_per_sec_display)

    # Initialize the spectrogram buffer
    S_buffer = np.full((len(env.freqs_plot), frames_per_window), -80.0)
    
    extent = [xaxis_extent[0], xaxis_extent[1], env.freqs_plot[0], env.freqs_plot[-1]]
    img = ax_spec.imshow(S_buffer, aspect='auto', origin='lower', 
                    cmap='magma', extent=extent, vmin=-80, vmax=0, animated=True)
                    
    ax_spec.set_xlim(xaxis_extent)
    title = ax_spec.set_title("Live Spectrogram")
    ax_spec.set_xlabel("Time (s, relative to now)")
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB")

    # Blitting setup
    fig.canvas.draw()
    plt.show(block=False)
    background = fig.canvas.copy_from_bbox(fig.bbox)
    
    while plt.fignum_exists(fig.number):
        # Observe one frame at a time
        frame, status = await env.observe()
        if frame is None:
            break # End of data
        
        # Translate the single frame to its dB representation
        frame_db = translate(frame, env)

        # Roll the buffer and add the new frame
        S_buffer = np.roll(S_buffer, -1, axis=1)
        S_buffer[:, -1] = frame_db
        
        # Update plot data dynamically
        img.set_data(S_buffer)
        
        buffer_size, buffer_percentage = status
        title.set_text(f"Live Spectrogram (T={current_time:.2f}s) | Buffer: {buffer_size}/{env.max_buffered_chunks} Frames ({buffer_percentage:.0f}%)")
        
        metrics_update_counter += 1
        if metrics_update_counter % 5 == 0: # Update metrics less frequently
            cpu_percent = psutil.cpu_percent()
            disk_io_now = psutil.disk_io_counters()
            if disk_io_now and disk_io_start:
                disk_read_diff = disk_io_now.read_bytes - disk_io_start.read_bytes
                disk_load_percent = min((disk_read_diff / (500 * 1024 * 1024)) * 100, 100)
            else:
                disk_load_percent = 0
            disk_io_start = disk_io_now
            
            process_count = len(psutil.pids())

            cpu_buffer = np.roll(cpu_buffer, -1)
            disk_buffer = np.roll(disk_buffer, -1)
            process_buffer = np.roll(process_buffer, -1)

            cpu_buffer[-1] = cpu_percent
            disk_buffer[-1] = disk_load_percent
            process_buffer[-1] = process_count

            line_cpu.set_ydata(cpu_buffer)
            line_disk.set_ydata(disk_buffer)
            line_proc.set_ydata(process_buffer)

            max_proc = max(process_buffer)
            if max_proc > 0:
                ax_metrics2.set_ylim(0, max(500, max_proc * 1.2))

        fig.canvas.restore_region(background)
        ax_spec.draw_artist(img)
        ax_spec.draw_artist(title)
        
        ax_metrics.draw_artist(line_cpu)
        ax_metrics.draw_artist(line_disk)
        ax_metrics2.draw_artist(line_proc)

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        
        # Calculate time advanced per frame
        time_per_frame = env.fft_hop_length / env.sr
        current_time += time_per_frame
        
        # No need for sleep, as observe() will wait for data
    plt.ioff()

if __name__ == "__main__":
    asyncio.run(main())