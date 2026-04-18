# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, and CQT Invariance
# =============================================================================

import librosa
import numpy as np
from librosa.feature.rhythm import tempo
from math import gcd
from functools import reduce

def find_optimal_lattice(onset_times, mean_ioi, tempo_val):
    if len(onset_times) < 2:
        return 0.001, 0.0
    # Use tempo to estimate initial base
    if tempo_val > 0:
        beat_period = 60 / tempo_val
        initial_base = beat_period / 64
    else:
        initial_base = mean_ioi / 16
    # Search for optimal divisor around estimated subdivisions
    possible_divisors = np.arange(8, 129, 4)  # Focus on multiples for rhythm subdivisions
    max_score = 0
    best_base = initial_base
    for d in possible_divisors:
        base = mean_ioi / d
        if base <= 0:
            continue
        quantized = np.round(onset_times / base) * base
        errors = np.abs(onset_times - quantized)
        mean_error = np.mean(errors)
        score = max(0, 1 - (mean_error / (base / 2)))  # Normalized score, higher better
        if score > max_score:
            max_score = score
            best_base = base
    return best_base, max_score

# List of available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    # Compute spectral centroid to classify sound type
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Adapt hop_length and onset params based on centroid for broad sound handling
    if centroid > 4000:
        print("  Detected high-centroid sound.")
        print("  Using high-sensitivity onset params.")
        hop_length = 128
        delta = 0.02
        wait = 1
    elif centroid > 1500:
        print("  Detected mid-centroid sound.")
        print("  Using mid-sensitivity onset params.")
        hop_length = 256
        delta = 0.05
        wait = 2
    else:
        print("  Detected low-centroid sound.")
        print("  Using low-sensitivity onset params.")
        hop_length = 512
        delta = 0.1
        wait = 4
    
    # Compute onset envelope with adaptive hop_length
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Compute tempo using the new import path to avoid warning
    tempo_val = tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Detect onsets with adaptive params
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time',
                                        hop_length=hop_length, delta=delta, wait=wait)
    print(f"  Detected onsets: {len(onsets)}")
    
    # Compute IOIs and rhythm coherence
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    std_ioi = np.std(iois) if len(iois) > 0 else 0
    rhythm_coherence = 1 - (std_ioi / mean_ioi) if mean_ioi > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice using optimal search with tempo hint
    lattice_base, lattice_coherence = find_optimal_lattice(onsets, mean_ioi, tempo_val)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Compute CQT with adaptive hop_length for improved invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Compute improved shift invariance metric
    if cqt.shape[1] > 1:
        shifts = []
        for i in range(cqt.shape[1] - 1):
            a = np.abs(cqt[:, i])
            b = np.abs(cqt[:, i + 1])
            corr = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            shifts.append(corr)
        invariance_metric = np.mean(shifts)
    else:
        invariance_metric = 0
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")