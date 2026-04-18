# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import find_peaks
from fractions import Fraction

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 4000:
        return 'high-centroid', {'backtrack': True, 'pre_max': 0.02, 'post_max': 0.02, 'delta': 0.05}
    elif mean_centroid > 1500:
        return 'mid-centroid', {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.1}
    else:
        return 'low-centroid', {'backtrack': True, 'pre_max': 0.1, 'post_max': 0.1, 'delta': 0.2}

# Improved rhythm lattice calculation using autocorrelation and peak clustering
def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Autocorrelation for better periodicity detection
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks, _ = find_peaks(autocorr, prominence=0.1 * np.max(autocorr))
    
    if len(peaks) > 1:
        peak_diffs = np.diff(peaks)
        base = np.gcd.reduce(peak_diffs.astype(int)) * (mean_ioi / len(iois))
    else:
        base = mean_ioi / 8  # Fallback to subdivision
    
    # Coherence: entropy-based regularity (lower entropy = higher coherence)
    hist, _ = np.histogram(iois, bins=20)
    hist = hist / hist.sum()
    coherence = 1 - entropy(hist) / np.log(20)
    
    # Lattice coherence: how well onsets fit multiples of base
    fits = np.min(np.abs(np.mod(onset_times[1:], base) / base))
    lattice_coherence = 1 - fits / len(onset_times)
    
    return mean_ioi, coherence, base, lattice_coherence

# Improved CQT with adaptive hop_length for better invariance
def compute_cqt_metrics(y, sr, centroid_class):
    # Adaptive hop_length: smaller for high-centroid to improve time resolution
    if centroid_class == 'high-centroid':
        hop_length = 256
    elif centroid_class == 'mid-centroid':
        hop_length = 512
    else:
        hop_length = 1024
    
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=672, bins_per_octave=84))
    
    # Shift invariance: compare with small time shift (e.g., 10 ms)
    shift_samples = int(0.01 * sr)
    y_shifted = np.roll(y, shift_samples)
    cqt_shifted = np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=hop_length, n_bins=672, bins_per_octave=84))
    
    # Align shapes if necessary
    min_frames = min(cqt.shape[1], cqt_shifted.shape[1])
    diff = np.mean(np.abs(cqt[:, :min_frames] - cqt_shifted[:, :min_frames]))
    max_val = np.max(cqt)
    invariance = diff / max_val if max_val > 0 else 0.0
    
    return cqt, invariance

# Main analysis function
def analyze_audio(filename):
    y, sr = librosa.load(filename, sr=22050)
    centroid_class, onset_params = classify_sound(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {centroid_class} sound.")
    print(f"  Using {centroid_class.split('-')[0]}-sensitivity onset params.")
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_lattice(onsets)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt, invariance = compute_cqt_metrics(y, sr, centroid_class)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")
    print()

# List of available files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Run analysis for each
for file in files:
    analyze_audio(file)