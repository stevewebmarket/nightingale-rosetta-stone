# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Sound Classification + Enhanced Lattice and Invariance
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce

def compute_rhythm_lattice(iois):
    if len(iois) < 2:
        return 0.05, 1.00
    iois_ms = [int(round(i * 1000)) for i in iois if i > 0]
    if not iois_ms:
        return 0.05, 1.00
    base_ms = reduce(gcd, iois_ms)
    base = max(base_ms / 1000.0, 0.01)  # Minimum base of 10ms
    # Coherence: fraction of IOIs that are multiples of base
    multiples = sum(1 for ioi in iois_ms if ioi % base_ms == 0) / len(iois_ms)
    return base, multiples

def compute_cqt_invariance(cqt):
    if cqt.shape[1] < 2:
        return 0.0
    # Improved metric: average normalized difference over small shifts (1-3 frames)
    diffs = []
    amp = np.mean(np.abs(cqt)) + 1e-8  # Avoid division by zero
    for shift in [1, 2, 3]:
        shifted = np.roll(cqt, shift, axis=1)
        diff = np.mean(np.abs(cqt[:, : -shift] - shifted[:, : -shift])) / amp
        diffs.append(diff)
    return np.mean(diffs)

def main():
    files = [f for f in os.listdir('.') if f.endswith('.wav')]
    if not files:
        print("No WAV files found, using synthetic test signal.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)  # A4 + A5 harmonic
        files = ['synthetic.wav']  # Placeholder for loop
        # Note: Synthetic not saved, processed in memory
    else:
        print("Analyzing available WAV files.")
    
    for file in files:
        if file == 'synthetic.wav':
            pass  # y, sr already defined
        else:
            y, sr = librosa.load(file, sr=22050)
        
        # Compute spectral centroid for sound classification
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = np.mean(centroid)
        
        if mean_centroid > 3000:  # High-centroid (e.g., birdsong)
            sound_type = "high-centroid sound (e.g., birdsong)"
            fmin = librosa.note_to_hz('C4')  # Start higher for birdsong
            bins_per_octave = 24  # Finer resolution for high frequencies
            n_bins = 120  # Extended bins for broader coverage
            onset_params = {'backtrack': False, 'pre_max': 0.02, 'post_max': 0.02}  # Sensitive for transient sounds
        elif mean_centroid > 1000:
            sound_type = "mid-centroid sound (e.g., orchestral or rock)"
            fmin = librosa.note_to_hz('C1')
            bins_per_octave = 12
            n_bins = 96
            onset_params = {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03}
        else:
            sound_type = "low-centroid sound (e.g., ambient or bass-heavy)"
            fmin = librosa.note_to_hz('C0')
            bins_per_octave = 12
            n_bins = 84
            onset_params = {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05}
        
        print(f"Analysis for {file}:")
        print(f"  Detected {sound_type}, using adjusted onset detection.")
        
        # Compute CQT with adaptive parameters
        cqt = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        
        # Onset detection with adaptive parameters
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
        print(f"  Detected onsets: {len(onset_times)}, ", end='')
        
        if len(onset_times) > 1:
            iois = np.diff(onset_times)
            mean_ioi = np.mean(iois)
            print(f"mean IOI: {mean_ioi:.2f} s, ", end='')
            # Rhythm coherence: 1 - coefficient of variation
            cv = np.std(iois) / (mean_ioi + 1e-8)
            rhythm_coherence = max(0, 1 - cv)
            print(f"rhythm coherence: {rhythm_coherence:.2f}")
        else:
            print("insufficient onsets")
            continue
        
        # Improved rhythm lattice
        base, lattice_coherence = compute_rhythm_lattice(iois)
        print(f"  Rhythm lattice base: {base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
        
        # Improved CQT shift invariance metric
        invariance = compute_cqt_invariance(cqt)
        print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

if __name__ == "__main__":
    main()