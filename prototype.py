# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Imports and Deprecations, Enhanced Synthetic Signal, Tempogram-based Rhythm Lattice
# =============================================================================

import librosa
import numpy as np
import scipy.signal

def generate_synthetic_test():
    sr = 22050
    duration = 10
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Base tone for harmonic content
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Harmonic for richer spectrum
    y += 0.3 * np.sin(2 * np.pi * 880 * t)
    # Rhythm component: clicks at ~120 BPM
    click_interval = 0.5  # seconds per beat
    click_times = np.arange(0, duration, click_interval)
    clicks = librosa.clicks(times=click_times, sr=sr, length=len(y), click_freq=800, click_duration=0.1)
    y += clicks
    # Broadband noise for varied sound handling
    noise = np.random.normal(0, 0.05, len(y))
    y += noise
    # Normalize to prevent clipping
    y /= np.max(np.abs(y)) + 1e-6
    return y, sr

def process_audio(y, sr):
    hop_length = 512
    # Improved CQT: log-scaled for better invariance to amplitude
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, fmin=librosa.note_to_hz('C1'))
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    print("CQT shape:", cqt_mag.shape)
    
    # Onset envelope with multi-band for better rhythm detection across frequencies
    onset_env = librosa.onset.onset_strength_multi(y=y, sr=sr, hop_length=hop_length, channels=[0, 1, 2, 3, 4], aggregate=np.median)
    onset_env = np.sum(onset_env, axis=0)  # Sum across bands for coherence
    
    # Tempo estimation using updated function
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print("Estimated tempo:", tempo)
    
    # Improved rhythm lattice: tempogram for rhythmic structure, averaged over time for coherence
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=384)
    rhythm_lattice = np.mean(tempogram, axis=1)  # Average over time for a 1D lattice
    # Additional smoothing for coherence
    rhythm_lattice = scipy.signal.medfilt(rhythm_lattice, kernel_size=3)
    print("Rhythm lattice shape:", rhythm_lattice.shape)

# Main execution
wav_files = []  # No files available
if not wav_files:
    print("No WAV files available, using synthetic test signal.")
    y, sr = generate_synthetic_test()
    process_audio(y, sr)
else:
    for file in wav_files:
        y, sr = librosa.load(file, sr=22050)
        process_audio(y, sr)