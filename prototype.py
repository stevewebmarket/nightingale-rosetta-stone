# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

def detect_sound_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1000:
        return "low-centroid sound"
    elif centroid < 4000:
        return "mid-centroid sound"
    else:
        return "high-centroid sound"

def compute_rhythm_metrics(onsets, sr):
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    if len(iois) == 0:
        return 0.0, 0.0, 0.010, 1.00  # Defaults for no onsets
    mean_ioi = np.mean(iois)
    # Improved coherence: normalized entropy of IOI histogram
    hist, _ = np.histogram(iois, bins=20)
    hist = hist / hist.sum()
    coherence = 1 - entropy(hist) / np.log(20)
    # Adaptive lattice: gcd of rounded IOIs in ms
    iois_ms = (iois * 1000).astype(int)
    if len(iois_ms) > 1:
        lattice_base = reduce(gcd, iois_ms) / 1000.0
    else:
        lattice_base = 0.010
    # Lattice coherence: fraction of onsets snapping to lattice
    snapped = np.round(times / lattice_base) * lattice_base
    errors = np.abs(times - snapped)
    lattice_coherence = np.mean(errors < (lattice_base / 2))
    return mean_ioi, coherence, lattice_base, lattice_coherence

def compute_cqt_invariance(cqt):
    # Improved invariance: average correlation across small shifts
    invariance = 0.0
    for shift in range(1, 4):  # Check small time shifts
        shifted = np.roll(cqt, shift, axis=1)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        invariance += corr
    invariance /= 3
    # Normalize to [0,1] with slight boost for broader handling
    invariance = (invariance + 1) / 2
    return invariance

def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    sound_type = detect_sound_type(y, sr)
    
    # Onset detection with adaptive backtracking for broader sounds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
    num_onsets = len(onsets)
    
    mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onsets, sr)
    
    # CQT with adjusted params for invariance and broad handling
    cqt = librosa.cqt(y, sr=sr, hop_length=256, n_bins=168, bins_per_octave=24)
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]
    invariance_metric = compute_cqt_invariance(np.abs(cqt))
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type}.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)