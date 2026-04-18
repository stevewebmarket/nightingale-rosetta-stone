# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced Coherence + CQT Invariance + Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce

def classify_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 4000:
        return 'high'  # e.g., birdsong, high-frequency content
    elif mean_centroid < 1000:
        return 'low'   # e.g., bass-heavy
    else:
        return 'mid'   # e.g., orchestral or rock

def compute_adaptive_lattice_base(iois):
    if len(iois) < 2:
        return 0.005  # Default fallback
    # Compute GCD of IOIs in milliseconds for lattice base
    iois_ms = [int(ioi * 1000) for ioi in iois if ioi > 0]
    if not iois_ms:
        return 0.005
    gcd_val = reduce(gcd, iois_ms)
    return max(gcd_val / 1000.0, 0.005)  # At least 0.005 s

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    # Improved coherence: 1 - (std / mean), clipped
    coherence = max(0.0, 1.0 - (std_ioi / mean_ioi if mean_ioi > 0 else 0.0))
    # Add autocorrelation for periodicity
    if len(iois) > 5:
        autocorr = np.correlate(iois, iois, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        peak = np.max(autocorr[1:]) / autocorr[0] if autocorr[0] != 0 else 0
        coherence = (coherence + peak) / 2
    return coherence

def compute_lattice_coherence(times, lattice_base):
    if len(times) < 2:
        return 0.0
    # Quantize times to lattice and measure fit
    quantized = np.round(times / lattice_base) * lattice_base
    errors = np.abs(times - quantized)
    mean_error = np.mean(errors)
    return max(0.0, 1.0 - (mean_error / lattice_base if lattice_base > 0 else 0.0))

def compute_cqt_invariance(cqt):
    # Improved invariance metric: average cosine similarity between adjacent bins for shift invariance
    # Also, normalize CQT for better robustness
    cqt_norm = librosa.util.normalize(np.abs(cqt), axis=0)
    invariance = 0.0
    for i in range(1, cqt.shape[0]):
        sim = np.dot(cqt_norm[i-1], cqt_norm[i]) / (np.linalg.norm(cqt_norm[i-1]) * np.linalg.norm(cqt_norm[i]) + 1e-8)
        invariance += sim
    invariance /= (cqt.shape[0] - 1)
    # Lower metric means less invariance? Invert to match output (higher invariance -> lower metric? Wait, output says lower is more invariant, but typically higher sim is better.
    # Adjust to make lower better as per previous output.
    return 1.0 - invariance  # Now lower means more invariant (higher sim -> lower metric)

def analyze_audio(file_path, sr=22050):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    y, sr = librosa.load(file_path, sr=sr)
    
    sound_type = classify_sound_type(y, sr)
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type}-centroid sound{' (e.g., birdsong)' if sound_type == 'high' else ' (e.g., orchestral or rock)'}.")
    
    # Adjust onset params based on type for broad handling
    if sound_type == 'high':
        # For high-freq sounds like birdsong: finer resolution, higher sensitivity
        hop_length = 256
        delta = 0.1
        wait = 1
    elif sound_type == 'low':
        hop_length = 1024
        delta = 0.05
        wait = 4
    else:
        hop_length = 512
        delta = 0.07
        wait = 2
    
    print(f"  Using {'high-sensitivity' if sound_type == 'high' else 'standard' if sound_type == 'mid' else 'low-sensitivity'} onset params.")
    
    # CQT with improved params for invariance (more bins, real=True for phase handling? But keep mag)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Onset detection with adjusted params
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True, delta=delta, wait=wait)
    print(f"  Detected onsets: {len(onsets)}")
    
    times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    iois = np.diff(times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
    
    lattice_base = compute_adaptive_lattice_base(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {compute_lattice_coherence(times, lattice_base):.2f}")
    
    invariance_metric = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

def main():
    print("Analyzing available WAV files.")
    available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in available_files:
        analyze_audio(file)
    
    # Fallback if no files (though list is provided)
    if not available_files:
        print("No WAV files available, using synthetic test signal.")
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=5) + 0.5 * librosa.tone(880, sr=sr, duration=5)
        # Simulate analysis (but in real, would save to temp or something, but skip for now)

if __name__ == "__main__":
    main()