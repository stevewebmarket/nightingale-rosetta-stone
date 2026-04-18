# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Hop Length + Rhythmic Lattice Optimization
# =============================================================================

import librosa
import numpy as np
import os
from functools import reduce
from math import gcd

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Compute spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 5000:
        sound_type = "high-centroid sound (e.g., birdsong)."
        sensitivity = "high-sensitivity"
        hop_length = 128
        delta = 0.05
    elif mean_centroid > 2000:
        sound_type = "mid-centroid sound (e.g., orchestral or rock)."
        sensitivity = "standard-sensitivity"
        hop_length = 256
        delta = 0.1
    else:
        sound_type = "low-centroid sound (e.g., bass-heavy)."
        sensitivity = "low-sensitivity"
        hop_length = 512
        delta = 0.2
    
    print(f"  Detected {sound_type}")
    print(f"  Using {sensitivity} onset params.")
    
    # Onset detection with adaptive hop_length
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True, delta=delta)
    
    # CQT with adaptive hop_length for improved invariance
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=144, bins_per_octave=24))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  Detected onsets: {len(onsets)}")
    
    # Compute IOIs and rhythm coherence
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 1 / (1 + cv) if mean_ioi > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Adaptive rhythm lattice calculation
    if len(onset_times) < 2:
        base = 0.1
    elif "high-centroid" in sound_type:
        # GCD-based for high-centroid (non-rhythmic)
        times_ms = np.round(onset_times * 1000).astype(int)
        diffs = np.diff(times_ms)
        base_ms = reduce(gcd, diffs) if len(diffs) > 0 else 100
        base = base_ms / 1000.0
    else:
        # Beat-tracking for mid/low-centroid (rhythmic)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_length)
        if len(beats) < 2 or tempo == 0:
            # Fallback to GCD
            times_ms = np.round(onset_times * 1000).astype(int)
            diffs = np.diff(times_ms)
            base_ms = reduce(gcd, diffs) if len(diffs) > 0 else 100
            base = base_ms / 1000.0
        else:
            beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            beat_iois = np.diff(beat_times)
            mean_beat_ioi = np.mean(beat_iois)
            base = mean_beat_ioi / 4  # Finer subdivision for lattice
    
    # Lattice coherence
    max_t = onset_times[-1] if len(onset_times) > 0 else 1
    lattice = np.arange(0, max_t + base, base)
    dists = [min(abs(t - l) for l in lattice) for t in onset_times]
    max_dist = base / 2
    lattice_coherence = np.mean(np.clip(1 - np.array(dists) / max_dist, 0, 1)) if len(dists) > 0 and max_dist > 0 else 0
    print(f"  Rhythm lattice base: {base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    # Improved CQT shift invariance metric (average over small shifts)
    metric = 0
    for shift in range(1, 4):  # Average over shifts 1-3 for better robustness
        shifted = np.roll(cqt, shift, axis=1)
        diff = np.abs(cqt[:, shift:] - shifted[:, shift:]) / (cqt[:, shift:] + 1e-6)
        metric += np.mean(diff)
    metric /= 3
    print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic test signals if no files
        print("No WAV files found. Generating synthetic test signals.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y_sine = np.sin(2 * np.pi * 440 * t)  # Simple sine wave
        np.save('synthetic_sine.npy', y_sine)  # Save for analysis, but skip actual analysis in this prototype
    else:
        for file in files:
            if os.path.exists(file):
                print(f"Analysis for {file}:")
                analyze_audio(file)
            else:
                print(f"{file} not found.")