# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Derivation + CQT Invariance Boost
# =============================================================================

import librosa
import numpy as np
import os
from scipy.stats import entropy
from scipy.signal import correlate
from math import gcd
from functools import reduce

# Constants
SR = 22050
CQT_N_BINS = 216  # Increased resolution for better invariance
CQT_BINS_PER_OCTAVE = 36  # Boosted for improved frequency shift invariance
CQT_HOP_LENGTH = 256  # Smaller hop for better time-shift invariance
CENTROID_THRESH_HIGH = 5000  # Hz, for high-centroid sounds like birdsong
CENTROID_THRESH_LOW = 2000   # Hz, for low-centroid sounds like bass-heavy

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def classify_sound(centroid):
    if centroid > CENTROID_THRESH_HIGH:
        return 'high'
    elif centroid < CENTROID_THRESH_LOW:
        return 'low'
    else:
        return 'mid'

def detect_onsets(y, sr, sound_type):
    if sound_type == 'high':
        # High-sensitivity for sparse, high-frequency onsets
        delta = 0.02
        wait = 1
        backtrack = True
    elif sound_type == 'low':
        # Low-sensitivity for dense, low-frequency onsets
        delta = 0.1
        wait = 4
        backtrack = False
    else:
        # Standard for mid-range
        delta = 0.07
        wait = 2
        backtrack = True
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, delta=delta, wait=wait, backtrack=backtrack, hop_length=512)
    return librosa.frames_to_time(onset_frames, sr=sr)

def compute_iois(onset_times):
    return np.diff(onset_times)

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    cv = np.std(iois) / np.mean(iois)
    return 1 / (1 + cv)  # Normalized coherence (higher when more regular)

def find_rhythm_lattice(iois):
    if len(iois) == 0:
        return 0.0, 0.0
    # Improved: Use autocorrelation to find dominant periodicity
    iois_rounded = np.round(iois, 2)  # Quantize to 0.01s
    autocorr = correlate(iois_rounded, iois_rounded, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = np.where(autocorr > np.max(autocorr) * 0.5)[0]
    if len(peaks) > 1:
        base = np.min(np.diff(peaks)) * 0.01  # Back to seconds
    else:
        base = np.gcd.reduce(np.round(iois * 100).astype(int)) / 100.0
    
    # Lattice coherence: entropy-based alignment
    quantized = np.round(iois / base)
    hist, _ = np.histogram(quantized, bins=np.arange(1, int(np.max(quantized))+2))
    hist = hist / np.sum(hist)
    latt_entropy = entropy(hist)
    max_entropy = np.log(len(hist))
    coherence = 1 - (latt_entropy / max_entropy if max_entropy > 0 else 0)
    
    return base, coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE)
    return np.abs(cqt)  # Magnitude for invariance

def cqt_shift_invariance(cqt):
    # Improved metric: Average correlation with small time-shifts
    corrs = []
    for shift in [1, 2, 3]:  # Small shifts in frames
        if cqt.shape[1] > shift:
            orig = cqt[:, :-shift].flatten()
            shifted = cqt[:, shift:].flatten()
            corr = np.corrcoef(orig, shifted)[0,1]
            corrs.append(corr)
    return 1 - np.mean(corrs) if corrs else 0.0  # Lower value means more invariant (less change)

def analyze_file(filename):
    print(f"Analysis for {filename}:")
    y, sr = librosa.load(filename, sr=SR)
    
    centroid = compute_spectral_centroid(y, sr)
    sound_type = classify_sound(centroid)
    print(f"  Detected {sound_type}-centroid sound (e.g., {'birdsong' if sound_type=='high' else 'orchestral or rock' if sound_type=='mid' else 'bass-heavy'}).")
    
    if sound_type == 'high':
        print("  Using high-sensitivity onset params.")
    elif sound_type == 'low':
        print("  Using low-sensitivity onset params.")
    else:
        print("  Using standard-sensitivity onset params.")
    
    onset_times = detect_onsets(y, sr, sound_type)
    print(f"  Detected onsets: {len(onset_times)}")
    
    iois = compute_iois(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        coh = rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coh:.2f}")
    else:
        print("  No IOIs detected.")
        mean_ioi, coh = 0.0, 0.0
    
    lattice_base, lattice_coh = find_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coh:.2f}")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {CQT_N_BINS}")
    
    inv_metric = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {inv_metric:.2f} (lower is more invariant)")

def main():
    print("Analyzing available WAV files.")
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    available_files = [f for f in wav_files if os.path.exists(f)]
    
    if not available_files:
        print("No WAV files found. Falling back to synthetic test signals.")
        # Synthetic signals would go here, but for now, skip
        return
    
    for filename in available_files:
        analyze_file(filename)

if __name__ == "__main__":
    main()