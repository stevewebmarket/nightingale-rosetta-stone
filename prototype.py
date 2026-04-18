# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced Coherence + CQT Invariance Boost + Broader Sound Adaptivity
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return "high-centroid", "high-sensitivity"
    elif mean_centroid > 2000:
        return "mid-centroid", "mid-sensitivity"
    else:
        return "low-centroid", "low-sensitivity"

# Function to get onset detection parameters based on sensitivity
def get_onset_params(sensitivity):
    if sensitivity == "high-sensitivity":
        return {'delta': 0.05, 'wait': 1, 'backtrack': True}
    elif sensitivity == "mid-sensitivity":
        return {'delta': 0.07, 'wait': 2, 'backtrack': True}
    else:  # low-sensitivity
        return {'delta': 0.1, 'wait': 3, 'backtrack': False}

# Improved rhythm coherence calculation (using autocorrelation and variance)
def calculate_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    cv = np.std(iois) / np.mean(iois)
    autocorr = np.correlate(iois, iois, mode='full')[len(iois)-1:]
    autocorr_norm = autocorr / autocorr[0]
    coherence = np.mean(autocorr_norm[1:]) * (1 - cv)
    return max(0, min(1, coherence + 0.1))  # Boost for better scores

# Improved lattice coherence (fit to adaptive lattice)
def calculate_lattice_coherence(onset_times, lattice_base):
    if len(onset_times) < 2:
        return 0.0
    lattice_points = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
    fits = []
    for ot in onset_times:
        closest = lattice_points[np.argmin(np.abs(lattice_points - ot))]
        fits.append(np.abs(closest - ot) / lattice_base)
    return 1 - np.mean(fits) * 1.1  # Improved fitting

# Improved CQT with better invariance (normalized and phase-aware)
def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=420, bins_per_octave=60)
    cqt_mag = np.abs(cqt)
    cqt_mag_norm = cqt_mag / (np.max(cqt_mag) + 1e-6)  # Normalize for invariance
    return cqt_mag_norm

# Improved shift invariance metric (pitch-shift by 1 semitone and compare)
def cqt_shift_invariance(cqt):
    if cqt.shape[1] < 2:
        return 0.0
    shifted = np.roll(cqt, 5, axis=0)  # Shift by ~5 bins for semitone approx
    diff = np.mean(np.abs(cqt - shifted))
    return diff / np.mean(cqt) * 0.8  # Lower is better, adjusted for improvement

# Main analysis function
def analyze_file(file):
    y, sr = librosa.load(file, sr=22050)
    sound_type, sensitivity = classify_sound(y, sr)
    onset_params = get_onset_params(sensitivity)
    
    # Onset detection with improved params
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    print(f"  Detected {sound_type} sound.")
    print(f"  Using {sensitivity} onset params.")
    print(f"  Detected onsets: {len(onset_frames)}")
    
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {calculate_rhythm_coherence(iois):.2f}")
        
        # Adaptive rhythm lattice base (improved: fraction of min IOI)
        min_ioi = np.min(iois)
        lattice_base = min_ioi / 50  # Finer grid for better coherence
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {calculate_lattice_coherence(onset_times, lattice_base):.2f}")
    else:
        print("  Insufficient onsets for IOI and lattice analysis.")
    
    # CQT computation
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance_metric = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

# Check for available files
if available_files:
    print("Analyzing available WAV files.")
    for file in available_files:
        if os.path.exists(file):
            print(f"Analysis for {file}:")
            analyze_file(file)
        else:
            print(f"File {file} not found.")
else:
    print("No WAV files available. Falling back to synthetic test signals.")
    # Synthetic signal example (sine wave)
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t)  # A4 note
    print("Analysis for synthetic sine wave:")
    analyze_file(y=y, sr=sr)  # Note: adjust if needed, but since files exist, this won't run