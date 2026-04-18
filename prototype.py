# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import os
import librosa
import numpy as np
from math import gcd
from functools import reduce
from scipy.spatial.distance import cosine

# Function to compute spectral centroid
def get_spectral_centroid(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(cent)

# Function to compute rounded GCD for rhythm lattice base
def compute_lattice_base(iois):
    if len(iois) < 2:
        return 0.0
    # Round to nearest 10 ms for robustness against floating-point precision
    iois_ms = np.round(iois * 100).astype(int)
    iois_ms = iois_ms[iois_ms > 0]  # Avoid zero or negative
    if len(iois_ms) == 0:
        return 0.0
    base_ms = reduce(gcd, iois_ms)
    return base_ms / 100.0

# Function to compute rhythm coherence (inverse of normalized std of IOIs)
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    return 1.0 / (1.0 + (std_ioi / mean_ioi)) if mean_ioi > 0 else 0.0

# Function to compute lattice coherence (how well onsets fit the lattice)
def compute_lattice_coherence(onset_times, lattice_base):
    if lattice_base <= 0 or len(onset_times) < 2:
        return 0.0
    # Project onsets to nearest lattice points
    phases = onset_times % lattice_base
    residuals = np.min(np.stack([phases, lattice_base - phases]), axis=0)
    normalized_residuals = residuals / lattice_base
    return 1.0 - np.mean(normalized_residuals)

# Function to compute CQT shift invariance metric
def compute_cqt_shift_invariance(y, sr, n_bins, fmin, hop_length=256):
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=fmin))
    # Shift by a small amount (e.g., 10 samples)
    y_shifted = np.roll(y, 10)
    cqt_shifted = np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=fmin))
    # Flatten and compute cosine distance (lower is more invariant)
    return cosine(cqt.flatten(), cqt_shifted.flatten())

# Main analysis function
def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    centroid = get_spectral_centroid(y, sr)
    
    # Detect sound type and adjust parameters
    if centroid > 5000:  # High-centroid (e.g., birdsong)
        print(f"  Detected high-centroid sound (e.g., birdsong), using adjusted onset detection.")
        onset_params = {'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.1}
        n_bins = 120
        fmin = librosa.note_to_hz('C3')  # Higher starting freq for high-centroid
        hop_length = 128  # Smaller hop for better invariance
    else:  # Mid/low-centroid (e.g., orchestral, rock)
        print(f"  Detected mid-centroid sound (e.g., orchestral or rock), using adjusted onset detection.")
        onset_params = {'pre_max': 0.01, 'post_max': 0.01, 'delta': 0.05}
        n_bins = 96
        fmin = librosa.note_to_hz('C1')  # Lower starting freq
        hop_length = 256  # Adjusted hop
    
    # Compute CQT with improved parameters (bins_per_octave=24 for better resolution)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, fmin=fmin)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    # Onset detection with backtracking for better accuracy
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    print(f"  Detected onsets: {len(onset_times)}", end='')
    
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        print(f", mean IOI: {mean_ioi:.2f} s", end='')
        rhythm_coherence = compute_rhythm_coherence(iois)
        print(f", rhythm coherence: {rhythm_coherence:.2f}")
        
        lattice_base = compute_lattice_base(iois)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s", end='')
        lattice_coherence = compute_lattice_coherence(onset_times, lattice_base)
        print(f", lattice coherence: {lattice_coherence:.2f}")
    else:
        print()
    
    # Compute improved CQT shift invariance
    invariance_metric = compute_cqt_shift_invariance(y, sr, n_bins, fmin, hop_length)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

# List available WAV files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

print("Analyzing available WAV files.")
for file in available_files:
    if os.path.exists(file):
        print(f"Analysis for {file}:")
        analyze_audio(file)
    else:
        print(f"File {file} not found.")