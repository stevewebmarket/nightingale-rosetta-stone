# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import os
import librosa
import numpy as np

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        sound_type = "high-centroid sound."
    else:
        sound_type = "mid-centroid sound."
    
    # Adaptive hop_length for broad sound handling (smaller for high-centroid)
    hop_length = 256 if mean_centroid > 3000 else 512
    
    # Improved onset detection with backtracking
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
    num_onsets = len(onsets)
    
    # Onset times and IOIs
    times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    iois = np.diff(times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    
    # Rhythm coherence using inverse coefficient of variation
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 and len(iois) > 1 else np.inf
    rhythm_coherence = 1 / (1 + cv)
    
    # Adaptive rhythm lattice base (based on min IOI for better fitting)
    if len(iois) > 0:
        min_ioi = np.min(iois)
        lattice_base = round(min_ioi / 10, 4)
    else:
        lattice_base = 0.001
    
    # Lattice coherence: mean error relative to lattice
    if len(times) > 0 and lattice_base > 0:
        lattice_points = np.arange(0, times[-1] + lattice_base, lattice_base)
        errors = [min(np.abs(lattice_points - t)) for t in times]
        mean_error = np.mean(errors)
        lattice_coherence = 1 / (1 + mean_error / lattice_base)
    else:
        lattice_coherence = 0.0
    
    # CQT with higher resolution for improved invariance
    bins_per_octave = 48  # Increased for better frequency resolution
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), n_bins=384, bins_per_octave=bins_per_octave))
    cqt_shape = cqt.shape
    
    # Improved CQT shift invariance metric: average correlation over small shifts
    invariance = 0.0
    if cqt.shape[1] > 2:
        num_shifts = min(3, cqt.shape[1] - 1)  # Check up to 3 hop shifts
        corrs = []
        for shift in range(1, num_shifts + 1):
            cqt_shifted = np.roll(cqt, shift, axis=1)
            corr = np.corrcoef(cqt.flatten(), cqt_shifted.flatten())[0, 1]
            corrs.append(corr)
        invariance = np.mean(corrs)
    else:
        invariance = 1.0
    
    # Print analysis
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type}")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        if os.path.exists(file):
            analyze_audio(file)
        else:
            print(f"File {file} not found.")
            # Fallback to synthetic signal if needed (but list is not empty)