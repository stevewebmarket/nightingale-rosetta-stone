# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Coherence + CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def classify_sound_type(centroid):
    if centroid < 1000:
        return "low-centroid sound"
    elif centroid > 5000:
        return "high-centroid sound"
    else:
        return "mid-centroid sound"

def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.001, 0.0  # Default fallback
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved lattice base: use GCD of quantized IOIs
    quantized_iois = np.round(iois * 1000).astype(int)  # Millisecond resolution
    if len(quantized_iois) == 0:
        lattice_base = 0.001
    else:
        lattice_base = reduce(gcd, quantized_iois) / 1000.0  # Back to seconds
    
    # Ensure reasonable base (not too small)
    if lattice_base < 0.001:
        lattice_base = mean_ioi / 4  # Fallback to quarter of mean IOI for subdivision
    
    # Lattice coherence: how well onsets fit the lattice (mean quantization error)
    quantized_onsets = np.round(onset_times / lattice_base) * lattice_base
    errors = np.abs(onset_times - quantized_onsets)
    coherence = 1 - (np.mean(errors) / lattice_base) if lattice_base > 0 else 0.0
    coherence = max(0.0, min(1.0, coherence))  # Clamp to [0,1]
    
    return lattice_base, coherence

def compute_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi if mean_ioi > 0 else 0.0
    coherence = 1 / (1 + cv)  # Improved: normalized coherence (higher for lower variation)
    return coherence

def compute_cqt_invariance(cqt, n_shifts=5):
    # Improved shift invariance: average correlation across small time shifts
    invariance = 0.0
    cqt_mag = np.abs(cqt)
    norm_cqt = cqt_mag / (np.linalg.norm(cqt_mag, axis=0, keepdims=True) + 1e-8)
    for shift in range(1, n_shifts + 1):
        corr = np.mean(np.sum(norm_cqt[:, :-shift] * norm_cqt[:, shift:], axis=0))
        invariance += corr
    invariance /= n_shifts
    return invariance

for file in wav_files:
    print(f"Analysis for {file}:")
    
    # Load audio
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid for classification (improved handling for broad sounds)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    sound_type = classify_sound_type(mean_centroid)
    print(f"  Detected {sound_type}.")
    
    # Onset detection (adapt backtracking based on sound type)
    backtrack = sound_type != "high-centroid sound"  # Less backtracking for high-frequency sounds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=backtrack, units='time')
    print(f"  Detected onsets: {len(onsets)}")
    
    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
    else:
        print("  mean IOI: 0.00 s, rhythm coherence: 0.00")
    
    # Rhythm lattice (improved)
    lattice_base, lattice_coherence = compute_rhythm_lattice(onsets)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT (adjusted parameters for better invariance, especially for natural sounds)
    min_freq = 32.7 if "birdsong" in file else 16.35  # Higher min for birdsong
    filter_scale = 1.5 if sound_type == "mid-centroid sound" else 1.0
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48,
                      fmin=min_freq, filter_scale=filter_scale)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # CQT shift invariance (improved metric)
    invariance_metric = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")