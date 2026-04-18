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

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        return "low-centroid"
    elif mean_centroid > 4000:
        return "high-centroid"
    else:
        return "mid-centroid"

# Function to get onset detection parameters based on sound type
def get_onset_params(sound_type):
    if sound_type == "low-centroid":
        return {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.1}
    elif sound_type == "high-centroid":
        return {'backtrack': False, 'pre_max': 0.02, 'post_max': 0.02, 'delta': 0.05}
    else:  # mid-centroid
        return {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.07}

# Improved rhythm lattice calculation using quantized IOIs and better GCD
def calculate_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    # Quantize IOIs to milliseconds for better GCD
    iois_ms = np.round(iois * 1000).astype(int)
    if len(iois_ms) == 0:
        return 0.0, 0.0
    base_ms = reduce(gcd, iois_ms)
    base_s = base_ms / 1000.0
    # Coherence: fraction of onsets aligning to lattice within tolerance
    tolerance = base_s * 0.1
    lattice_points = np.arange(onset_times[0], onset_times[-1] + base_s, base_s)
    alignments = 0
    for ot in onset_times:
        if min(abs(ot - lattice_points)) <= tolerance:
            alignments += 1
    coherence = alignments / len(onset_times)
    return base_s, coherence

# Improved coherence calculation using entropy of IOI histogram
def calculate_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    hist, _ = np.histogram(iois, bins=20)
    hist = hist / hist.sum()
    ent = entropy(hist)
    max_ent = np.log(20)  # Max entropy for uniform dist
    return 1 - (ent / max_ent)  # Higher regularity -> higher coherence

# Improved CQT with normalization for better shift invariance
def compute_cqt_invariance(y, sr):
    # Use higher resolution CQT with 48 bins per octave over 8 octaves
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=384, bins_per_octave=48)
    # Normalize magnitudes for invariance
    cqt_mag = np.abs(cqt)
    cqt_mag_norm = cqt_mag / (np.linalg.norm(cqt_mag, axis=0, keepdims=True) + 1e-8)
    # Shift invariance: average correlation with octave shifts
    invariance = 0.0
    for shift in [12, 24]:  # 1 and 2 octaves
        shifted = np.roll(cqt_mag_norm, shift, axis=0)
        corr = np.mean(np.abs(np.corrcoef(cqt_mag_norm.flatten(), shifted.flatten()))[0, 1])
        invariance += corr
    invariance /= 2
    return cqt, invariance

# Main analysis function
def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    sound_type = classify_sound(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type} sound.")
    
    onset_params = get_onset_params(sound_type)
    print(f"  Using {'low' if sound_type == 'low-centroid' else 'high' if sound_type == 'high-centroid' else 'mid'}-sensitivity onset params.")
    
    # Onset detection with adaptive parameters
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times) if len(onset_times) > 1 else []
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {calculate_rhythm_coherence(iois):.2f}")
    
    lattice_base, lattice_coherence = calculate_rhythm_lattice(onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt, invariance = compute_cqt_invariance(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# List of available files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Run analysis for each file
for file in files:
    analyze_audio(file)
    print()  # Blank line between analyses