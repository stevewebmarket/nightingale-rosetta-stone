# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from fractions import Fraction
import math

# Function to compute greatest common divisor for floats with tolerance
def float_gcd(numbers, tolerance=1e-4):
    def gcd(a, b):
        while abs(b) > tolerance:
            a, b = b, a % b
        return a
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd(result, num)
    return result

# Function to compute rhythm coherence using autocorrelation
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    # Normalize IOIs
    iois_norm = iois / np.mean(iois)
    # Autocorrelation-like measure
    autocorr = np.correlate(iois_norm, iois_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    # Coherence as max autocorrelation beyond zero lag, normalized
    if len(autocorr) > 1:
        coherence = np.max(autocorr[1:]) / autocorr[0]
    else:
        coherence = 0.0
    return round(coherence, 2)

# Function to compute adaptive rhythm lattice base
def compute_rhythm_lattice(iois, sr):
    if len(iois) < 2:
        return 1.0 / sr, 1.00  # Default to sample rate resolution
    # Compute GCD of IOIs
    gcd_ioi = float_gcd(iois)
    # Quantize to nearest multiple of 1/sr
    lattice_base = max(round(gcd_ioi * sr) / sr, 1.0 / sr)
    # Lattice coherence: fraction of IOIs that fit the lattice
    fits = np.sum(np.isclose(iois % lattice_base, 0, atol=1e-3) | np.isclose(iois % lattice_base, lattice_base, atol=1e-3))
    coherence = fits / len(iois)
    return lattice_base, round(coherence, 2)

# Enhanced CQT with shift invariance metric
def compute_cqt_and_invariance(y, sr, n_bins=384):
    # Use CQT with higher hop_length for better temporal resolution
    hop_length = 256  # Increased resolution
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=48))  # Higher bins_per_octave for finer freq resolution
    # Shift invariance: compute correlation between original and shifted CQT
    if cqt.shape[1] > 1:
        shift = 1  # Small shift
        cqt_shifted = np.roll(cqt, shift, axis=1)
        # Normalize
        cqt_norm = cqt / (np.linalg.norm(cqt) + 1e-8)
        cqt_shifted_norm = cqt_shifted / (np.linalg.norm(cqt_shifted) + 1e-8)
        # Cosine similarity as invariance metric
        invariance = np.mean(np.diag(np.dot(cqt_norm.T, cqt_shifted_norm), k=0))
    else:
        invariance = 1.0
    # Improve invariance for broad sounds by averaging with STFT for mid/low freqs
    if invariance < 0.7:  # Threshold for enhancement
        stft = np.abs(librosa.stft(y, hop_length=hop_length))
        # Hybrid: replace lower bins with STFT
        hybrid_cqt = np.vstack((stft[:n_bins//2, :], cqt[n_bins//2:, :]))
        # Recompute invariance on hybrid
        hybrid_norm = hybrid_cqt / (np.linalg.norm(hybrid_cqt) + 1e-8)
        hybrid_shifted = np.roll(hybrid_cqt, shift, axis=1)
        hybrid_shifted_norm = hybrid_shifted / (np.linalg.norm(hybrid_shifted) + 1e-8)
        invariance = np.mean(np.diag(np.dot(hybrid_norm.T, hybrid_shifted_norm), k=0))
    return cqt, round(invariance, 2)

# Main analysis function
def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        sound_type = "high-centroid sound."
    else:
        sound_type = "mid-centroid sound."
    
    # Onset detection with backtracking for broader sound handling
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, pre_max=0.05, post_max=0.05)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onsets_count = len(onset_times)
    
    # Inter-onset intervals (IOIs)
    if onsets_count > 1:
        iois = np.diff(onset_times)
        mean_ioi = round(np.mean(iois), 2)
        rhythm_coherence = compute_rhythm_coherence(iois)
    else:
        mean_ioi = 0.0
        rhythm_coherence = 0.0
    
    # Adaptive rhythm lattice
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois if onsets_count > 1 else [], sr)
    lattice_base = round(lattice_base, 3)
    
    # Enhanced CQT and invariance
    cqt, invariance = compute_cqt_and_invariance(y, sr)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type}")
    print(f"  Detected onsets: {onsets_count}")
    print(f"  mean IOI: {mean_ioi} s, rhythm coherence: {rhythm_coherence}")
    print(f"  Rhythm lattice base: {lattice_base} s")
    print(f"  lattice coherence: {lattice_coherence}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance} (higher is more invariant)")

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Analyze each file
for wav in wav_files:
    analyze_audio(wav)