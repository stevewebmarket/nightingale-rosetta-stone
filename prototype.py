# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

# Function to detect spectral centroid category
def detect_centroid_category(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid sound"
    elif mean_centroid > 1000:
        return "mid-centroid sound"
    else:
        return "low-centroid sound"

# Improved rhythm lattice calculation
def calculate_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    # Use GCD of quantized IOIs for base lattice, improved with finer quantization
    quantized_iois = np.round(iois * 1000).astype(int)  # Millisecond quantization
    lattice_base = reduce(gcd, quantized_iois) / 1000.0
    # Improved coherence: entropy-based, lower entropy means higher coherence
    hist, _ = np.histogram(iois, bins=20)
    hist = hist / hist.sum()
    coherence = 1 - entropy(hist) / np.log(20)  # Normalized entropy
    return mean_ioi, coherence, lattice_base

# Improved CQT with better shift invariance
def compute_cqt_invariance(y, sr):
    # Use higher bins_per_octave for better frequency resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.5)
    # Normalize CQT for shift invariance
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    cqt_norm = (cqt_mag - np.min(cqt_mag)) / (np.max(cqt_mag) - np.min(cqt_mag))
    # Shift invariance metric: correlation between original and shifted versions
    shifts = [1, 2, 3]  # Check invariance to small bin shifts
    invariance_scores = []
    for shift in shifts:
        shifted = np.roll(cqt_norm, shift, axis=0)
        corr = np.corrcoef(cqt_norm.flatten(), shifted.flatten())[0, 1]
        invariance_scores.append(corr)
    return cqt.shape, np.mean(invariance_scores)

# Main analysis function
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    category = detect_centroid_category(y, sr)
    
    # Onset detection with improved sensitivity for broad sounds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, units='time')
    num_onsets = len(onsets)
    
    # Rhythm metrics
    mean_ioi, rhythm_coherence, lattice_base = calculate_rhythm_lattice(onsets)
    
    # Lattice coherence: improved by checking alignment to base
    if lattice_base > 0:
        alignments = np.mod(onsets, lattice_base)
        lattice_coherence = 1 - np.std(alignments) / lattice_base
    else:
        lattice_coherence = 0.0
    
    # CQT invariance
    cqt_shape, invariance_metric = compute_cqt_invariance(y, sr)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {category}.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Analyze each file
for file in wav_files:
    analyze_audio(file)