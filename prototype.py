# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

# Function to compute spectral centroid category
def get_centroid_category(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1000:
        return 'low'
    elif centroid < 4000:
        return 'mid'
    else:
        return 'high'

# Function to detect onsets with sensitivity based on category
def detect_onsets(y, sr, category):
    if category == 'low':
        hop_length = 512
        delta = 0.1
        wait = 4
    elif category == 'mid':
        hop_length = 256
        delta = 0.05
        wait = 2
    else:
        hop_length = 128
        delta = 0.02
        wait = 1
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, delta=delta, wait=wait)
    return librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

# Function to compute rhythm coherence (std dev of IOIs normalized)
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    return 1 / (1 + std_ioi / mean_ioi) if mean_ioi > 0 else 0.0

# Function to compute rhythm lattice base using GCD of quantized IOIs
def compute_rhythm_lattice(iois):
    if len(iois) < 2:
        return 0.001
    # Quantize to milliseconds
    quantized_iois = [int(round(ioi * 1000)) for ioi in iois if ioi > 0]
    if not quantized_iois:
        return 0.001
    lattice_base_ms = reduce(gcd, quantized_iois)
    return max(lattice_base_ms / 1000.0, 0.001)  # Convert back to seconds, min 1ms

# Function to compute lattice coherence (fraction of onsets near lattice points)
def compute_lattice_coherence(onset_times, lattice_base):
    if len(onset_times) < 2 or lattice_base <= 0:
        return 0.0
    # Generate lattice points up to duration
    duration = onset_times[-1]
    lattice_points = np.arange(0, duration + lattice_base, lattice_base)
    # For each onset, find min distance to lattice point
    hits = 0
    tolerance = lattice_base * 0.1  # 10% tolerance
    for ot in onset_times:
        distances = np.abs(lattice_points - ot)
        if np.min(distances) <= tolerance:
            hits += 1
    return hits / len(onset_times)

# Function to compute CQT with improved parameters
def compute_cqt(y, sr):
    return librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.5)

# Function to compute CQT shift invariance (avg correlation with shifts)
def cqt_shift_invariance(cqt, max_shift=5):
    cqt_mag = np.abs(cqt)
    rows, cols = cqt_mag.shape
    if cols < max_shift * 2:
        return 0.0
    correlations = []
    for shift in range(1, max_shift + 1):
        orig = cqt_mag[:, :cols - shift]
        shifted = cqt_mag[:, shift:]
        corr = np.corrcoef(orig.flatten(), shifted.flatten())[0, 1]
        correlations.append(corr)
    return np.mean(correlations)

# Main analysis function
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    category = get_centroid_category(y, sr)
    print(f"Analysis for {file_path}:")
    print(f"  Detected {category}-centroid sound.")
    print(f"  Using {category}-sensitivity onset params.")
    
    onset_times = detect_onsets(y, sr, category)
    print(f"  Detected onsets: {len(onset_times)}")
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
    
    lattice_base = compute_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {compute_lattice_coherence(onset_times, lattice_base):.2f}")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {cqt_shift_invariance(cqt):.2f} (higher is more invariant)")
    print()

# List of available files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Run analysis for each
for file in files:
    analyze_audio(file)