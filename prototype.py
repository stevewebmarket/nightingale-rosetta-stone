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
from scipy.spatial.distance import cosine
from math import gcd
from functools import reduce

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 4000:
        return 'high'
    elif mean_centroid > 1500:
        return 'mid'
    else:
        return 'low'

def detect_onsets(y, sr, centroid_type):
    if centroid_type == 'high':
        hop_length = 256
        backtrack = True
        pre_post_max = 5
    elif centroid_type == 'mid':
        hop_length = 512
        backtrack = True
        pre_post_max = 3
    else:
        hop_length = 1024
        backtrack = False
        pre_post_max = 2
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, 
                                        backtrack=backtrack, pre_max=pre_post_max, post_max=pre_post_max)
    return librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

def compute_iois(onset_times):
    return np.diff(onset_times)

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, _ = np.histogram(iois, bins=50)
    return 1 - entropy(hist) / np.log(len(hist))

def improved_rhythm_lattice(iois):
    if len(iois) < 2:
        return 0.0
    iois_ms = iois * 1000
    iois_ms = iois_ms[iois_ms > 5]  # Filter out very small IOIs (noise)
    if len(iois_ms) < 2:
        return 0.0
    gcd_val = reduce(gcd, iois_ms.astype(int))
    cluster_centers = np.percentile(iois_ms, [25, 50, 75])
    base = np.min(cluster_centers[cluster_centers > 0]) / 1000
    return max(gcd_val / 1000, base / 4)  # Ensure reasonable minimum

def lattice_coherence(iois, base):
    if base == 0:
        return 0.0
    multiples = iois / base
    residuals = multiples - np.round(multiples)
    return 1 - np.mean(np.abs(residuals))

def compute_cqt(y, sr):
    return librosa.cqt(y, sr=sr, hop_length=512, n_bins=84*8, bins_per_octave=84, filter_scale=1.0)

def cqt_shift_invariance(cqt):
    mag = np.abs(cqt)
    if mag.shape[1] < 2:
        return 0.0
    shifts = []
    for i in range(1, min(10, mag.shape[1])):
        shift_mag = np.roll(mag, i, axis=1)
        sim = 1 - cosine(mag.flatten(), shift_mag.flatten())
        shifts.append(sim)
    invariance = 1 - np.mean(shifts)  # Invert to make lower better
    return np.clip(invariance, 0, 1)

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=22050)
    centroid_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {centroid_type}-centroid sound.")
    print(f"  Using {centroid_type}-sensitivity onset params.")
    
    onset_times = detect_onsets(y, sr, centroid_type)
    print(f"  Detected onsets: {len(onset_times)}")
    
    iois = compute_iois(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
        
        lattice_base = improved_rhythm_lattice(iois)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence(iois, lattice_base):.2f}")
    else:
        print("  No IOIs detected.")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    analyze_file(file)
    print()