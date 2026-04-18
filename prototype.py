# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice and CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return 'high'
    elif mean_centroid > 2000:
        return 'mid'
    else:
        return 'low'

def detect_onsets(y, sr, sensitivity):
    if sensitivity == 'high':
        hop_length = 256
        backtrack = True
        energy = librosa.feature.rms(y=y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    elif sensitivity == 'mid':
        hop_length = 512
        backtrack = False
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    else:
        hop_length = 1024
        backtrack = False
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=backtrack)
    return onsets

def compute_iois(onset_times):
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
    return mean_ioi, max(0, coherence)

def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.001, 0.0
    iois = np.diff(onset_times)
    iois = iois[iois > 0]
    if len(iois) == 0:
        return 0.001, 0.0
    # Improved: use GCD of quantized IOIs for better lattice base
    quantized_iois = np.round(iois * 1000).astype(int)  # to ms
    base_ms = reduce(gcd, quantized_iois)
    base = base_ms / 1000.0 if base_ms > 0 else 0.001
    # Coherence: fraction of onsets fitting the lattice
    lattice_points = np.arange(onset_times[0], onset_times[-1] + base, base)
    fits = np.sum([np.any(np.isclose(lp, onset_times, atol=base/2)) for lp in lattice_points])
    coherence = fits / len(lattice_points) if len(lattice_points) > 0 else 0
    return base, coherence

def compute_cqt(y, sr):
    # Improved: use smaller hop_length for better time resolution, and HCQT-like for invariance
    cqt = librosa.hybrid_cqt(y, sr=sr, n_bins=384, bins_per_octave=48, hop_length=256, fmin=librosa.note_to_hz('C1'))
    return cqt

def cqt_shift_invariance(cqt, shift=1):
    # Improved: compute normalized correlation for better invariance metric
    if cqt.shape[1] < shift + 1:
        return 0.0
    cqt_norm = cqt / (np.linalg.norm(cqt, axis=0, keepdims=True) + 1e-8)
    corr = np.mean(np.abs(np.dot(cqt_norm.T, cqt_norm).diagonal(offset=shift)))
    # Entropy-based invariance
    mag = np.abs(cqt)
    shifted = np.roll(mag, shift, axis=1)
    kl_div = entropy(mag.flatten() + 1e-8, shifted.flatten() + 1e-8)
    invariance = corr / (1 + kl_div) if kl_div > 0 else corr
    return invariance

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    y, sr = librosa.load(file, sr=22050)
    centroid_level = compute_spectral_centroid(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_level}-centroid sound.")
    sensitivity = 'high' if centroid_level == 'high' else 'mid' if centroid_level == 'mid' else 'low'
    print(f"  Using {sensitivity}-sensitivity onset params.")
    onsets = detect_onsets(y, sr, sensitivity)
    print(f"  Detected onsets: {len(onsets)}")
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    mean_ioi, rhythm_coherence = compute_iois(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print()