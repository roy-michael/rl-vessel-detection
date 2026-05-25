import os
import glob
import numpy as np
import librosa
from sklearn.decomposition import NMF

def diagnose():
    file_dir = "D:/RoyStudies/Recordings/Croatia/Ocean Sonics/2507_1"
    wav_files = sorted(glob.glob(os.path.join(file_dir, "*.wav")))
    if not wav_files:
        print("No wav files found.")
        return
    
    file_path = wav_files[0]
    print(f"Loading {file_path}...")
    
    # Load 10 seconds of audio to analyze
    y, sr = librosa.load(file_path, sr=None, duration=10.0)
    print(f"Sample rate: {sr} Hz, signal shape: {y.shape}")
    print(f"Signal mean: {np.mean(y):.6f}, std: {np.std(y):.6f}, min: {np.min(y):.6f}, max: {np.max(y):.6f}")
    
    # STFT settings from main/agent
    n_fft = 32768
    hop_length = n_fft // 16 # 2048
    
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    print(f"STFT shape: {S.shape}")
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Let's check energy distribution across frequencies
    print("\nEnergy distribution by frequency range:")
    for f_low, f_high in [(0, 10), (10, 50), (50, 100), (100, 500), (500, 2000), (2000, sr/2)]:
        mask = (freqs >= f_low) & (freqs < f_high)
        energy = np.sum(S[mask, :]**2)
        print(f"  {f_low}-{f_high} Hz: energy={energy:.2e}, number of bins={np.sum(mask)}")
        
    # Let's crop to 0-2000 Hz as in main
    min_freq = 0
    max_freq = 2000
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    S_input = S[freq_mask, :]
    
    # Fit NMF on raw magnitude
    print("\nFitting NMF (k=5) on raw magnitude...")
    model = NMF(n_components=5, init='nndsvda', solver='cd', max_iter=2000, random_state=42)
    W = model.fit_transform(S_input)
    H = model.components_
    
    for i in range(5):
        h = H[i, :]
        print(f"  Component {i} Activation - Mean: {np.mean(h):.2e}, Var: {np.var(h):.2e}, Max: {np.max(h):.2e}")

    # Now let's try with high-pass filtering (removing very low frequencies, e.g. < 10 Hz)
    print("\nFitting NMF (k=5) on magnitude with min_freq = 10 Hz...")
    freq_mask_10hz = (freqs >= 10) & (freqs <= max_freq)
    S_input_10hz = S[freq_mask_10hz, :]
    model_10hz = NMF(n_components=5, init='nndsvda', solver='cd', max_iter=2000, random_state=42)
    W_10hz = model_10hz.fit_transform(S_input_10hz)
    H_10hz = model_10hz.components_
    for i in range(5):
        h = H_10hz[i, :]
        print(f"  Component {i} (>=10Hz) - Mean: {np.mean(h):.2e}, Var: {np.var(h):.2e}, Max: {np.max(h):.2e}")

    # Let's check with PCEN or log scaling
    print("\nFitting NMF (k=5) on log-scaled magnitude (amplitude to db)...")
    S_db = librosa.amplitude_to_db(S_input, ref=np.max(S_input))
    # Since NMF needs non-negative values, we shift S_db to be >= 0 (e.g. S_db - min(S_db))
    S_db_shifted = S_db - np.min(S_db)
    model_db = NMF(n_components=5, init='nndsvda', solver='cd', max_iter=2000, random_state=42)
    W_db = model_db.fit_transform(S_db_shifted)
    H_db = model_db.components_
    for i in range(5):
        h = H_db[i, :]
        print(f"  Component {i} (dB scaled) - Mean: {np.mean(h):.2e}, Var: {np.var(h):.2e}, Max: {np.max(h):.2e}")

if __name__ == "__main__":
    diagnose()
