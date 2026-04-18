# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence + CQT Shift Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import find_peaks
from math import gcd
from functools import reduce

# Constants
SR = 22050
CQT_N_BINS = 384  # Increased for better frequency resolution
CQT_FMIN = librosa.note_to_hz('C0')  # Lower fmin for broader range
CQT_BINS_PER_OCTAVE = 48  # Higher for improved invariance
HOP_LENGTH = 256  # Smaller hop for finer time resolution and better invariance

# Available files
FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 500:
        return 'low'
    elif mean_centroid < 2000:
        return 'mid'
    else:
        return 'high'

def get_onset_params(sound_type):
    if sound_type == 'low':
        return {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.2}
    elif sound_type == 'mid':
        return {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.1}
    else:
        return {'backtrack': True, 'pre_max': 0.02, 'post_max': 0.02, 'delta': 0.05}

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, bin_edges = np.histogram(iois, bins=20)
    hist = hist / hist.sum()
    return 1 - entropy(hist) / np.log(len(hist))

def improved_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    # Use autocorrelation for better base period estimation
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks, _ = find_peaks(autocorr, prominence=0.1 * autocorr.max())
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        base = np.median(peak_diffs) * np.mean(iois) / len(iois)
    else:
        base = np.gcd.reduce(iois.astype(int)) / 1000.0 if np.all(iois > 0) else np.mean(iois)
    # Lattice coherence: how well onsets align to multiples of base
    alignments = np.abs(onset_times % base)
    coherence = 1 - np.mean(alignments) / (base / 2)
    return base, max(coherence, 0.0)

def compute_cqt_invariance(cqt, n_shifts=5):
    # Improved metric: average cosine similarity across small time shifts
    similarities = []
    for shift in range(1, n_shifts + 1):
        shifted = np.roll(cqt, shift, axis=1)
        # Cosine similarity per frequency bin
        norm_orig = np.linalg.norm(cqt, axis=1)
        norm_shift = np.linalg.norm(shifted, axis=1)
        dot = np.sum(cqt * shifted, axis=1)
        sim = np.mean(dot / (norm_orig * norm_shift + 1e-8))
        similarities.append(sim)
    return np.mean(similarities)

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=SR)
    sound_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type}-centroid sound.")
    params = get_onset_params(sound_type)
    print(f"  Using {sound_type}-sensitivity onset params.")
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, **params)
    print(f"  Detected onsets: {len(onsets)}")
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP_LENGTH)
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        coherence = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coherence:.2f}")
        lattice_base, lattice_coh = improved_rhythm_lattice(onset_times)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coh:.2f}")
    else:
        print("  Insufficient onsets for rhythm analysis.")
    # CQT with improved params for invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE))
    print(f"  CQT shape: {cqt.shape}, n_bins: {CQT_N_BINS}")
    inv_metric = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {inv_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    for file in FILES:
        analyze_file(file)