# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice and CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce
import os

def compute_gcd_base(iois):
    if len(iois) < 2:
        return np.mean(iois)
    # Round to milliseconds for practical GCD computation
    iois_ms = np.round(iois * 1000).astype(int)
    # Avoid zero or negative
    iois_ms = iois_ms[iois_ms > 0]
    if len(iois_ms) < 2:
        return np.mean(iois)
    gcd_val = reduce(gcd, iois_ms)
    return gcd_val / 1000.0

def normalize_frames(spec):
    norms = np.linalg.norm(spec, axis=0, keepdims=True)
    norms[norms == 0] = 1
    return spec / norms

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        sound_type = "low-centroid sound."
    elif mean_centroid > 4000:
        sound_type = "high-centroid sound."
    else:
        sound_type = "mid-centroid sound."
    
    # Improved onset detection with backtracking and delta
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True, delta=0.1)
    num_onsets = len(onset_times)
    
    if num_onsets < 2:
        print(f"  Insufficient onsets detected for {file}.")
        return
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved rhythm coherence using normalized entropy of IOI histogram
    hist, _ = np.histogram(iois, bins=30)
    hist = hist / (hist.sum() + 1e-12)
    ent = entropy(hist)
    max_ent = np.log(30)
    rhythm_coherence = 1 - (ent / max_ent)
    
    # Improved rhythm lattice base using approximate GCD with tempo fallback
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if tempo > 0:
        beat_duration = 60 / tempo
        lattice_base = min(compute_gcd_base(iois), beat_duration / 4)  # Subdivide for lattice
    else:
        lattice_base = compute_gcd_base(iois)
    lattice_base = round(lattice_base, 3)
    
    # Improved lattice coherence: variance of onsets modulo base
    if lattice_base > 0:
        mods = onset_times % lattice_base
        normalized_var = np.var(mods) / (lattice_base ** 2 / 12)  # Uniform var normalization
        lattice_coherence = 1 / (1 + normalized_var)
        lattice_coherence = round(max(0, min(1, lattice_coherence)), 2)
    else:
        lattice_coherence = 0.0
    
    # CQT with improved parameters for better invariance (higher bins_per_octave, smaller hop)
    n_bins = 384
    bins_per_octave = 64  # Increased for finer frequency resolution
    hop_length = 128  # Smaller hop for better time resolution
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave))
    
    # CQT shift invariance metric with normalization for improvement
    shift_samples = sr // 20  # Smaller shift for metric (0.05s)
    y_shifted = np.roll(y, shift_samples)
    cqt_shifted = np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave))
    
    cqt_norm = normalize_frames(cqt)
    cqt_shifted_norm = normalize_frames(cqt_shifted)
    
    min_len = min(cqt_norm.shape[1], cqt_shifted_norm.shape[1])
    corr_matrix = np.corrcoef(cqt_norm[:, :min_len].flatten(), cqt_shifted_norm[:, :min_len].flatten())
    invariance = round(corr_matrix[0, 1], 2)
    
    # Print results
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type}")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# List of available files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Fallback to synthetic if empty (though not needed here)
if not available_files:
    # Synthetic test signal
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    np.save('synthetic.npy', y)  # Not a WAV, but for demo
    analyze_audio('synthetic.npy')  # Adjust load if needed
else:
    for file in available_files:
        if os.path.exists(file):
            analyze_audio(file)
        else:
            print(f"File {file} not found, skipping.")