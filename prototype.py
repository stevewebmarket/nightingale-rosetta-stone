# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, and CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce

def classify_sound_type(centroid):
    if centroid > 5000:
        return 'high-centroid'
    elif centroid > 2000:
        return 'mid-centroid'
    else:
        return 'low-centroid'

def get_onset_params(sound_type):
    if sound_type == 'high-centroid':
        return {'pre_max': 0.01, 'post_max': 0.01, 'backtrack': True}
    elif sound_type == 'mid-centroid':
        return {'pre_max': 0.05, 'post_max': 0.05, 'backtrack': True}
    else:
        return {'pre_max': 0.1, 'post_max': 0.1, 'backtrack': True}

def compute_rhythm_lattice_base(iois):
    if len(iois) < 2:
        return 0.001
    # Round to milliseconds for GCD computation
    iois_ms = np.round(iois * 1000).astype(int)
    base_ms = reduce(gcd, iois_ms)
    return base_ms / 1000.0

def compute_lattice_coherence(iois, base):
    if base == 0 or len(iois) == 0:
        return 0.0
    multiples = iois / base
    errors = np.abs(multiples - np.round(multiples))
    return np.mean(errors < 1e-2)

def compute_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    mean_ioi = np.mean(iois)
    if mean_ioi == 0:
        return 0.0
    return 1 - (np.std(iois) / mean_ioi)

def get_cqt_hop_length(sound_type, sr, mean_ioi):
    base_hop = 512
    if sound_type == 'high-centroid':
        return max(128, int(sr * mean_ioi / 4))  # Smaller for high detail
    elif sound_type == 'mid-centroid':
        return base_hop
    else:
        return min(1024, int(sr * mean_ioi * 2))  # Larger for stability

def compute_cqt_shift_invariance(cqt):
    if cqt.shape[1] < 6:
        return 0.0
    norm = np.mean(np.abs(cqt)) + 1e-8
    diff = 0.0
    for shift in [1, 2, 3]:
        shifted = np.roll(cqt, shift, axis=1)
        diff += np.mean(np.abs(cqt[:, :-shift] - shifted[:, :-shift])) / norm
    return diff / 3

def generate_synthetic_signal(sr, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return y

def main():
    sr = 22050
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    available_files = [f for f in wav_files if os.path.exists(f)]
    
    if not available_files:
        print("No WAV files found. Using synthetic test signal.")
        y = generate_synthetic_signal(sr)
        filename = 'synthetic.wav'
        available_files = [filename]  # Placeholder for printing
    else:
        print("Analyzing available WAV files.")
    
    for filename in available_files:
        if filename == 'synthetic.wav':
            continue  # Already generated y
        else:
            y, _ = librosa.load(filename, sr=sr)
        
        print(f"Analysis for {filename}:")
        
        # Spectral centroid
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sound_type = classify_sound_type(centroid)
        print(f"  Detected {sound_type} sound.")
        
        # Onset detection
        params = get_onset_params(sound_type)
        print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
        onsets = librosa.onset.onset_detect(y=y, sr=sr, **params)
        print(f"  Detected onsets: {len(onsets)}")
        
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
        
        # Rhythm lattice
        lattice_base = compute_rhythm_lattice_base(iois)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        lattice_coherence = compute_lattice_coherence(iois, lattice_base)
        print(f"  lattice coherence: {lattice_coherence:.2f}")
        
        # CQT with adaptive hop
        hop_length = get_cqt_hop_length(sound_type, sr, mean_ioi)
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=252, bins_per_octave=36)
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        
        invariance_metric = compute_cqt_shift_invariance(np.abs(cqt))
        print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

if __name__ == "__main__":
    main()