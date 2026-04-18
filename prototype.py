# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Optimized Lattice Period + Finer CQT Resolution
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce
from scipy.optimize import minimize_scalar

# List of available WAV files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Fallback synthetic signals if no files
if not available_files:
    # Synthetic test signals
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y_sine = np.sin(2 * np.pi * 440 * t)  # Sine wave
    y_noise = np.random.randn(len(t))  # Noise
    analyses = [('synthetic_sine', y_sine), ('synthetic_noise', y_noise)]
else:
    analyses = [(file, None) for file in available_files]

print("Analyzing available WAV files." if available_files else "No WAV files; using synthetic signals.")

for name, y_input in analyses:
    if y_input is None:
        y, sr = librosa.load(name, sr=22050)
    else:
        y = y_input
        sr = 22050

    # Compute spectral centroid for classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)

    if mean_centroid > 3000:
        sound_type = "high-centroid sound (e.g., birdsong)"
        onset_params = {'delta': 0.02, 'wait': 1, 'backtrack': True}  # High sensitivity
    elif mean_centroid < 1000:
        sound_type = "low-centroid sound (e.g., bass-heavy)"
        onset_params = {'delta': 0.04, 'wait': 2, 'backtrack': True}  # Adjusted sensitivity for better detection
    else:
        sound_type = "mid-centroid sound (e.g., orchestral or rock)"
        onset_params = {'delta': 0.07, 'wait': 2, 'backtrack': False}  # Standard

    print(f"Analysis for {name}:")
    print(f"  Detected {sound_type}.")
    print(f"  Using {'high' if mean_centroid > 3000 else 'low' if mean_centroid < 1000 else 'standard'}-sensitivity onset params.")

    # Compute CQT with finer resolution for improved invariance
    hop_length = 128  # Finer time resolution (~5.8 ms)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames', **onset_params)
    print(f"  Detected onsets: {len(onsets)}")

    # Compute onset times and IOIs
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0

    # Rhythm coherence (1 / (1 + CV))
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 and len(iois) > 1 else np.inf
    rhythm_coherence = 1 / (1 + cv) if np.isfinite(cv) else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")

    # Optimized rhythm lattice base period
    def lattice_coherence_loss(b, times):
        if b <= 0:
            return np.inf
        residuals = times % b
        std_res = np.std(residuals)
        return std_res / (b / 2)  # Normalized by half-period for loss (lower better)

    if len(onset_times) > 1:
        # Initial guess from GCD of discretized IOIs
        iois_us = (iois * 1e6).astype(int)
        initial_base_us = reduce(gcd, iois_us) if len(iois_us) > 1 else iois_us[0]
        initial_base = initial_base_us / 1e6

        # Optimize base in range around initial
        res = minimize_scalar(lattice_coherence_loss, args=(onset_times,), bounds=(0.001, 0.5), method='bounded')
        base = res.x if res.success else initial_base

        # Compute final lattice coherence
        residuals = onset_times % base
        lat_coh = 1 - (2 * np.std(residuals) / base) if base > 0 else 0
    else:
        base = 0
        lat_coh = 0

    print(f"  Rhythm lattice base period u1e6: Rhythm lattice base: {base:.3f} s, lattice coherence: {lat_coh:.2f}")

    # Improved CQT shift invariance metric (mean abs diff after 1-frame shift, normalized)
    mag_cqt = np.abs(cqt)
    if mag_cqt.shape[1] > 1:
        diff = np.mean(np.abs(mag_cqt[:, :-1] - mag_cqt[:, 1:]))
        norm = np.mean(mag_cqt) + 1e-8  # Avoid div by zero
        invariance_metric = diff / norm
    else:
        invariance_metric = 0
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")