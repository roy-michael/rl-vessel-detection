import numpy as np
import soundfile as sf
import os

def generate_test_wav(filepath="test_signal.wav", sr=22050, duration=5.0):
    """
    Generates a WAV file with 5 segments of 1 second each:
    - 0-1s: Quiet sine wave (amplitude 0.1)
    - 1-2s: Medium sine wave (amplitude 0.5)
    - 2-3s: Silence (amplitude 0.0)
    - 3-4s: Loud sine wave (amplitude 0.9)
    - 4-5s: Linear fade out from 0.8 to 0.0
    """
    print(f"Generating test audio file: {filepath}")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base frequency of 440 Hz
    frequency = 440.0
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Apply amplitudes to each 1-second segment
    # Each segment is exactly sr samples long
    samples_per_sec = sr
    
    # Segment 1 (0-1s): amplitude 0.1
    signal[0:samples_per_sec] *= 0.1
    
    # Segment 2 (1-2s): amplitude 0.5
    signal[samples_per_sec:2*samples_per_sec] *= 0.5
    
    # Segment 3 (2-3s): silence (0.0)
    signal[2*samples_per_sec:3*samples_per_sec] *= 0.0
    
    # Segment 4 (3-4s): amplitude 0.9
    signal[3*samples_per_sec:4*samples_per_sec] *= 0.9
    
    # Segment 5 (4-5s): fade out from 0.8 to 0.0
    fade = np.linspace(0.8, 0.0, samples_per_sec)
    signal[4*samples_per_sec:5*samples_per_sec] *= fade

    # Ensure filepath is absolute or relative to workspace
    sf.write(filepath, signal, sr)
    print(f"Test audio saved successfully. File size: {os.path.getsize(filepath)} bytes.")
    print(f"Expected metrics for each 1-second segment (pure sine waves RMS = Amplitude / sqrt(2) approx. Amplitude * 0.707):")
    print("  Segment 1 (0-1s): Max Abs = 0.1, RMS approx. 0.0707")
    print("  Segment 2 (1-2s): Max Abs = 0.5, RMS approx. 0.3535")
    print("  Segment 3 (2-3s): Max Abs = 0.0, RMS = 0.0000")
    print("  Segment 4 (3-4s): Max Abs = 0.9, RMS approx. 0.6364")
    print("  Segment 5 (4-5s): Max Abs = 0.8 (starts at 0.8), RMS is mathematically lower due to fade")

if __name__ == "__main__":
    generate_test_wav()
