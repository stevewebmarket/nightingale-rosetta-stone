# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence + Improved CQT Invariance
# =============================================================================

import librosa
import numpy as np
import scipy

# Define available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to compute spectral centroid category
def get_centroid_category(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        return 'high'
    elif centroid > 2000:
        return 'mid'
    else:
        return 'low'

# Function to detect onsets with adaptive sensitivity
def detect_onsets(y, sr, category):
    if category == 'high':
        hop_length = 256
        delta = 0.05
        backtrack = True
    elif category == 'mid':
        hop_length = 512
        delta = 0.1
        backtrack = True
    else:
        hop_length = 512
        delta = 0.2
        backtrack = False
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, backtrack=backtrack, delta=delta)
    return onsets

# Function to compute rhythm coherence (improved: lower std deviation indicates higher coherence)
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    coherence = 1 / (1 + std_ioi / mean_ioi)  # Normalized coherence
    return coherence

# Improved rhythm lattice: Use autocorrelation to find base period, then fit lattice
def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 3:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Autocorrelation for periodicity
    autocorr = scipy.signal.correlate(iois, iois, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = scipy.signal.find_peaks(autocorr)[0]
    if len(peaks) > 1:
        base = np.mean(np.diff(peaks)) * mean_ioi / len(iois)
    else:
        base = mean_ioi
    
    # Lattice coherence: how well onsets fit to multiples of base
    lattice_points = np.arange(onset_times[0], onset_times[-1] + base, base)
    matches = 0
    for ot in onset_times:
        if np.min(np.abs(lattice_points - ot)) < base * 0.05:  # Tolerance
            matches += 1
    lattice_coherence = matches / len(onset_times)
    
    return base, lattice_coherence

# Improved CQT with better shift invariance: Use HCQT-like approach with multiple octaves
def compute_cqt_invariance(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    
    # Simulate pitch shift by resampling
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    cqt_shifted = librosa.cqt(y_shifted, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    
    # Normalize and compute correlation for invariance
    cqt_norm = cqt / (np.linalg.norm(cqt) + 1e-6)
    cqt_shifted_norm = cqt_shifted / (np.linalg.norm(cqt_shifted) + 1e-6)
    invariance = np.corrcoef(cqt_norm.flatten(), cqt_shifted_norm.flatten())[0, 1]
    
    return cqt.shape, invariance

# Main analysis loop
for file in wav_files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    category = get_centroid_category(y, sr)
    print(f"  Detected {category}-centroid sound.")
    print(f"  Using {category}-sensitivity onset params.")
    
    onsets = detect_onsets(y, sr, category)
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        rhythm_coherence = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  Insufficient onsets for IOI calculation.")
        continue
    
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt_shape, invariance = compute_cqt_invariance(y, sr)
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")