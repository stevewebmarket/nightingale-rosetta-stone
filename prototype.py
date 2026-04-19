# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence and CQT Shift Invariance
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Detect spectral centroid for sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        centroid_type = "high-centroid"
        fmin = librosa.note_to_hz('C3')
        n_bins = 192
    elif mean_centroid > 2000:
        centroid_type = "mid-centroid"
        fmin = librosa.note_to_hz('C2')
        n_bins = 168
    else:
        centroid_type = "low-centroid"
        fmin = librosa.note_to_hz('C1')
        n_bins = 168
    
    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    num_onsets = len(onsets)
    
    if num_onsets < 2:
        mean_ioi = 0.0
        rhythm_coherence = 0.0
        lattice_base = 0.001
        lattice_coherence = 0.0
    else:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        rhythm_coherence = 1 / (1 + std_ioi / (mean_ioi + 1e-10))
        
        # Improved rhythm lattice base using GCD with rounding to 10ms precision
        iois_ms = [int(round(ioi * 100)) for ioi in iois if ioi > 0]
        if iois_ms:
            lattice_base_ms = reduce(gcd, iois_ms)
            if lattice_base_ms == 0:
                lattice_base_ms = 1
            lattice_base = lattice_base_ms / 100.0
        else:
            lattice_base = 0.001
        
        # Lattice coherence: fraction of onsets fitting the lattice within tolerance
        if lattice_base > 0:
            quantized = np.round(onsets / lattice_base)
            errors = np.abs(quantized * lattice_base - onsets)
            lattice_coherence = np.mean(errors < (lattice_base / 2))
        else:
            lattice_coherence = 0.0
    
    # CQT with improved parameters for invariance (higher bins per octave)
    bins_per_octave = 24
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_shape = cqt.shape
    
    # Improved CQT shift invariance metric (average similarity after 1-bin roll)
    cqt_mag = np.abs(cqt)
    cqt_norm = cqt_mag / (np.linalg.norm(cqt_mag, axis=0, keepdims=True) + 1e-10)
    shifted = np.roll(cqt_norm, 1, axis=0)
    sims = np.mean(cqt_norm * shifted, axis=0)
    invariance = np.mean(sims)
    
    # Print results
    print(f"  Detected {centroid_type} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        print(f"Analysis for {file}:")
        analyze_audio(file)