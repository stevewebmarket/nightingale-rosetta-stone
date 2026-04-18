# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice + Enhanced Invariance
# =============================================================================

import os
import numpy as np
import librosa

# Constants
SR = 22050
CQT_BINS_PER_OCTAVE = 36  # Increased for finer resolution
CQT_N_OCTAVES = 7
CQT_N_BINS = CQT_BINS_PER_OCTAVE * CQT_N_OCTAVES  # 252 bins now
HOP_LENGTH = 512

# Available WAV files
AVAILABLE_WAVS = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Synthetic signal fallback if no files
def generate_synthetic_signal():
    t = np.linspace(0, 5, 5 * SR, endpoint=False)
    freqs = np.array([220, 440, 880])
    signal = np.sum(np.sin(2 * np.pi * freqs[:, np.newaxis] * t), axis=0)
    signal += 0.5 * np.random.randn(len(t))  # Add noise
    return signal / np.max(np.abs(signal))

# Classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 4000:
        return 'high'  # e.g., birdsong
    elif centroid < 1000:
        return 'low'  # e.g., bass-heavy
    else:
        return 'mid'  # e.g., orchestral or rock

# Adaptive onset detection parameters
def get_onset_params(sound_type):
    if sound_type == 'high':
        return {'backtrack': True, 'delta': 0.05, 'wait': 1}  # High sensitivity
    elif sound_type == 'low':
        return {'backtrack': False, 'delta': 0.2, 'wait': 4}  # Low sensitivity
    else:
        return {'backtrack': True, 'delta': 0.1, 'wait': 2}  # Standard

# Improved rhythm lattice: adaptive base and coherence
def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    rhythm_coherence = 1 / (1 + np.std(iois) / mean_ioi)  # Normalized coherence
    
    # Adaptive lattice base: fraction of mean IOI
    lattice_base = mean_ioi / 20  # Finer grid
    lattice_multiples = np.arange(1, 21) * lattice_base
    hits = np.sum([np.min(np.abs(iois - m)) < (lattice_base / 2) for m in lattice_multiples])
    lattice_coherence = hits / len(iois)
    
    return len(onset_times), mean_ioi, rhythm_coherence, lattice_base, lattice_coherence

# Enhanced CQT with improved shift invariance (using HCQT-like approach)
def compute_cqt_metrics(y, sr):
    # Compute HCQT for better invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE, fmin=librosa.note_to_hz('C1'))
    
    # Simulate shift by rolling and compute difference
    cqt_shifted = np.roll(cqt, shift=1, axis=1)
    diff = np.mean(np.abs(cqt - cqt_shifted)) / np.mean(np.abs(cqt))
    
    return cqt.shape, CQT_N_BINS, diff

# Main analysis function
def analyze_audio(file_path=None, use_synthetic=False):
    if use_synthetic:
        y = generate_synthetic_signal()
        print("Using synthetic signal.")
    else:
        y, _ = librosa.load(file_path, sr=SR)
        print(f"Analysis for {os.path.basename(file_path)}:")
    
    sound_type = classify_sound(y, SR)
    print(f"  Detected {sound_type}-centroid sound.")
    onset_params = get_onset_params(sound_type)
    print(f"  Using {sound_type}-sensitivity onset params.")
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP_LENGTH, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP_LENGTH)
    
    n_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onset_times)
    print(f"  Detected onsets: {n_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    cqt_shape, n_bins, invariance_metric = compute_cqt_metrics(y, SR)
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

# Run analysis
print("Analyzing available WAV files.")
for wav in AVAILABLE_WAVS:
    if os.path.exists(wav):
        analyze_audio(wav)
    else:
        print(f"File {wav} not found, skipping.")

if not AVAILABLE_WAVS:
    analyze_audio(use_synthetic=True)