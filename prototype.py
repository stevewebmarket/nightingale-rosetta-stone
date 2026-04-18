# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, and CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import mode
from math import gcd
from functools import reduce

def compute_spectral_centroid(y, sr):
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroids)

def classify_sound_type(centroid):
    if centroid < 1500:
        return 'low-centroid'
    elif centroid < 4000:
        return 'mid-centroid'
    else:
        return 'high-centroid'

def get_onset_params(sound_type):
    if sound_type == 'low-centroid':
        return {'pre_max': 0.02, 'post_max': 0.02, 'wait': 0.02}  # Low sensitivity
    elif sound_type == 'mid-centroid':
        return {'pre_max': 0.03, 'post_max': 0.03, 'wait': 0.03}  # Mid sensitivity
    else:
        return {'pre_max': 0.04, 'post_max': 0.04, 'wait': 0.04}  # High sensitivity

def compute_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
    return 1 / (1 + cv)

def compute_rhythm_lattice(iois, tol=0.01):
    if len(iois) < 2:
        return 0.001, 0.0
    # Generate candidate bases from unique IOIs divided by 1 to 8
    unique_iois = np.unique(np.round(iois, 3))
    candidates = []
    for ioi in unique_iois:
        for div in range(1, 9):
            candidates.append(ioi / div)
    candidates = np.unique(np.round(candidates, 4))
    # Add GCD-based candidate
    scaled_iois = (iois * 10000).astype(int)
    gcd_val = reduce(gcd, scaled_iois) / 10000.0
    candidates = np.append(candidates, gcd_val)
    candidates = np.unique(candidates)
    
    best_coherence = 0.0
    best_base = 0.001
    for cand in candidates:
        if cand <= 0:
            continue
        fits = 0
        for ioi in iois:
            k = round(ioi / cand)
            if abs(ioi - k * cand) < tol:
                fits += 1
        coh = fits / len(iois)
        if coh > best_coherence:
            best_coherence = coh
            best_base = cand
    return best_base, best_coherence

def compute_cqt_invariance(cqt, shift_frames=1):
    if cqt.shape[1] < shift_frames + 1:
        return 0.0
    # Compute average cosine similarity between frames and shifted frames
    similarities = []
    for i in range(cqt.shape[1] - shift_frames):
        frame1 = cqt[:, i]
        frame2 = cqt[:, i + shift_frames]
        sim = np.dot(frame1, frame2) / (np.linalg.norm(frame1) * np.linalg.norm(frame2) + 1e-8)
        similarities.append(sim)
    return np.mean(similarities)

def analyze_audio(filename, sr=22050):
    y, sr = librosa.load(filename, sr=sr)
    centroid = compute_spectral_centroid(y, sr)
    sound_type = classify_sound_type(centroid)
    print(f"  Detected {sound_type} sound.")
    onset_params = get_onset_params(sound_type)
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        rhythm_coh = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}")
        
        base, lattice_coh = compute_rhythm_lattice(iois)
        print(f"  Rhythm lattice base: {base:.3f} s")
        print(f"  lattice coherence: {lattice_coh:.2f}")
    else:
        print("  Insufficient onsets for rhythm analysis.")
    
    # Improved CQT with smaller hop_length for better time resolution and invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=256, n_bins=672, bins_per_octave=84))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic signals
        sr = 22050
        y_sine = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5 * sr))  # 5s sine wave
        analyze_audio_signal(y_sine, sr, "synthetic_sine")
        # Add more if needed
    else:
        for file in files:
            print(f"Analysis for {file}:")
            analyze_audio(file)

def analyze_audio_signal(y, sr, name):
    # Similar to analyze_audio but for in-memory signal
    centroid = compute_spectral_centroid(y, sr)
    sound_type = classify_sound_type(centroid)
    print(f"  Detected {sound_type} sound for {name}.")
    # Etc. (omitted for brevity, adapt as needed)

if __name__ == "__main__":
    main()