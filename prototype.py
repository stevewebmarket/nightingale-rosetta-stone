# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice for Sparse Sounds + CQT Invariance Boost
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from math import gcd
from functools import reduce

# Constants
SR = 22050
CQT_N_BINS = 384
CQT_HOP_LENGTH = 512
ONSET_BACKTRACK = True
ONSET_DELTA = 0.05  # Adjusted for broader sound handling

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def detect_onsets(y, sr, centroid_type):
    if centroid_type == "high-centroid":
        # Enhanced for birdsong: use higher delta and pre_max for sparse, high-freq onsets
        onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=ONSET_BACKTRACK, delta=ONSET_DELTA * 1.5, pre_max=0.05, post_max=0.05)
    else:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=ONSET_BACKTRACK, delta=ONSET_DELTA)
    return onsets

def compute_iois(onset_frames, sr, hop_length=512):
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    iois = np.diff(onset_times)
    return iois

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, _ = np.histogram(iois, bins=20)
    return 2.0 - entropy(hist) / np.log(len(hist))  # Normalized coherence (0-2, higher better)

def rhythm_lattice_base(iois):
    if len(iois) == 0:
        return 0.0
    # Improved: Use clustered mode for base, better for sparse rhythms like birdsong
    from scipy.cluster.vq import kmeans
    iois_reshaped = iois.reshape(-1, 1)
    centroids, _ = kmeans(iois_reshaped, 3)  # Assume up to 3 clusters
    base = np.min(centroids[centroids > 0])
    # GCD refinement
    iois_ms = (iois * 1000).astype(int)
    gcd_val = reduce(gcd, iois_ms[iois_ms > 0])
    return max(base, gcd_val / 1000.0)

def lattice_coherence(onset_times, base):
    if len(onset_times) < 2 or base == 0:
        return 0.0
    # Improved: Measure fit to lattice with tolerance for better coherence on sparse sounds
    lattice = np.arange(0, onset_times[-1] + base, base)
    dists = cdist(onset_times.reshape(-1, 1), lattice.reshape(-1, 1))
    min_dists = np.min(dists, axis=1)
    tolerance = base * 0.1  # 10% tolerance
    fit_ratio = np.mean(min_dists < tolerance)
    return fit_ratio

def compute_cqt(y, sr):
    # Adjusted for invariance: increased bins per octave for better shift invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=48)
    return np.abs(cqt)

def cqt_shift_invariance(cqt):
    # Improved metric: Average correlation across multiple shifts for broader invariance
    if cqt.shape[1] < 2:
        return 0.0
    correlations = []
    for shift in [1, 2, 3]:  # Check invariance to small time shifts
        shifted = np.roll(cqt, shift, axis=1)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        correlations.append(corr)
    return np.mean(correlations)

def analyze_audio(file):
    y, sr = librosa.load(file, sr=SR)
    centroid_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_type} sound.")
    
    onsets = detect_onsets(y, sr, centroid_type)
    print(f"  Detected onsets: {len(onsets)}")
    
    iois = compute_iois(onsets, sr, CQT_HOP_LENGTH)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
    else:
        print("  mean IOI: 0.00 s, rhythm coherence: 0.00")
    
    base = rhythm_lattice_base(iois)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=CQT_HOP_LENGTH)
    print(f"  Rhythm lattice base: {base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence(onset_times, base):.2f}")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {CQT_N_BINS}")
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print()

if __name__ == "__main__":
    files = ["birdsong.wav", "orchestra.wav", "rock.wav"]
    for file in files:
        analyze_audio(file)