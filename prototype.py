# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce

print("Analyzing available WAV files.")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)

    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_cent = np.mean(centroid)

    if mean_cent < 2000:
        sound_type = "low-centroid sound."
        sensitivity = "low"
    elif mean_cent < 5000:
        sound_type = "mid-centroid sound."
        sensitivity = "mid"
    else:
        sound_type = "high-centroid sound."
        sensitivity = "high"

    print(f"  Detected {sound_type}")
    
    # Adjust onset detection parameters based on sensitivity
    if sensitivity == "low":
        delta = 0.1
        wait = 1
        pre_max = 0.03
        post_max = 0.03
    elif sensitivity == "mid":
        delta = 0.05
        wait = 2
        pre_max = 0.05
        post_max = 0.05
    else:
        delta = 0.02
        wait = 3
        pre_max = 0.07
        post_max = 0.07

    print(f"  Using {sensitivity}-sensitivity onset params.")

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', delta=delta, wait=wait,
                                        pre_max=pre_max, post_max=post_max, backtrack=True)
    print(f"  Detected onsets: {len(onsets)}")

    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s", end="")

        # Rhythm coherence: inverse of coefficient of variation (improved for broader handling)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv) * (1 - np.exp(-len(onsets)/100))  # Adjust for onset density
        print(f", rhythm coherence: {rhythm_coherence:.2f}")

        # Improved rhythm lattice: gcd of quantized IOIs with fallback and min base
        iois_ms = np.round(iois * 1000).astype(int)
        iois_ms = iois_ms[iois_ms > 0]  # Avoid zero
        if len(iois_ms) > 1:
            base_ms = reduce(gcd, iois_ms)
            base = max(base_ms / 1000.0, 0.001)  # Min base 1ms
        else:
            base = mean_ioi / 4 if mean_ioi > 0 else 0.001  # Fallback subdivision

        print(f"  Rhythm lattice base: {base:.3f} s")

        # Improved lattice coherence: fit quality with tolerance
        lattice_points = np.arange(0, onsets[-1] + base, base)
        dists = []
        for o in onsets:
            min_dist = min(abs(o - lp) for lp in lattice_points)
            dists.append(min_dist)
        mean_dist = np.mean(dists)
        tol = base * 0.1  # 10% tolerance for better coherence on varied sounds
        lattice_coherence = np.mean(np.array(dists) < tol)  # Fraction within tolerance
        print(f"  lattice coherence: {lattice_coherence:.2f}")

    else:
        print("  Insufficient onsets for rhythm analysis.")
        continue

    # Improved CQT: higher bins_per_octave for better shift invariance, normalized
    cqt = librosa.cqt(y, sr=sr, hop_length=256, n_bins=252, bins_per_octave=36, filter_scale=1.5)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    print(f"  CQT shape: {cqt_mag.shape}, n_bins: {cqt_mag.shape[0]}")

    # Improved invariance metric: average normalized diff over small shifts (1-3 bins)
    diffs = []
    for shift in [1, 2, 3]:
        shifted = np.roll(cqt_mag, shift, axis=0)
        diff = np.mean(np.abs(cqt_mag - shifted)) / (np.mean(np.abs(cqt_mag)) + 1e-6)
        diffs.append(diff)
    invariance_metric = np.mean(diffs)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")