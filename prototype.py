# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import os
import numpy as np
import librosa

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def detect_onsets(y, sr, sensitivity='mid'):
    if sensitivity == 'high':
        hop_length = 256
        backtrack = True
        pre_max = 0.02
        post_max = 0.02
    elif sensitivity == 'mid':
        hop_length = 512
        backtrack = True
        pre_max = 0.03
        post_max = 0.03
    else:
        hop_length = 1024
        backtrack = False
        pre_max = 0.05
        post_max = 0.05
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length,
                                        backtrack=backtrack, pre_max=pre_max * sr / hop_length,
                                        post_max=post_max * sr / hop_length)
    return onsets

def compute_iois(onset_times):
    return np.diff(onset_times)

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    return 1 / (1 + np.std(iois) / np.mean(iois))

def improved_rhythm_lattice(iois):
    if len(iois) == 0:
        return 0.0, 0.0
    from math import gcd
    from functools import reduce
    iois_ms = iois * 1000
    gcd_ms = reduce(gcd, iois_ms.astype(int))
    base = gcd_ms / 1000.0
    if base == 0:
        base = np.min(iois)
    ratios = iois / base
    coherence = 1 / (1 + np.mean(np.abs(ratios - np.round(ratios))))
    return base, coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=252, bins_per_octave=36, filter_scale=1.0)
    return np.abs(cqt)

def cqt_shift_invariance(cqt, sr, shift_samples=10):
    y_shifted = np.roll(cqt, shift_samples, axis=1)
    diff = np.mean(np.abs(cqt - y_shifted)) / np.mean(np.abs(cqt))
    return diff

def analyze_file(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    centroid = compute_spectral_centroid(y, sr)
    
    if centroid > 5000:
        sensitivity = 'high'
        print(f"  Detected high-centroid sound.")
    elif centroid > 2000:
        sensitivity = 'mid'
        print(f"  Detected mid-centroid sound.")
    else:
        sensitivity = 'low'
        print(f"  Detected low-centroid sound.")
    
    print(f"  Using {sensitivity}-sensitivity onset params.")
    
    onsets = detect_onsets(y, sr, sensitivity)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    print(f"  Detected onsets: {len(onsets)}")
    
    iois = compute_iois(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
    else:
        print("  No IOIs detected.")
        return
    
    lattice_base, lattice_coherence = improved_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    invariance = cqt_shift_invariance(cqt, sr)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

def generate_synthetic_signal(sr=22050, duration=10.0):
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return y

def main():
    print("Analyzing available WAV files.")
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if not wav_files:
        print("No WAV files found. Falling back to synthetic test signal.")
        y = generate_synthetic_signal()
        sr = 22050
        file_name = "synthetic.wav"
        # Simulate analysis
        centroid = compute_spectral_centroid(y, sr)
        sensitivity = 'mid' if 2000 < centroid <= 5000 else 'high' if centroid > 5000 else 'low'
        print(f"Analysis for {file_name}:")
        print(f"  Detected {'high' if sensitivity=='high' else 'mid' if sensitivity=='mid' else 'low'}-centroid sound.")
        print(f"  Using {sensitivity}-sensitivity onset params.")
        onsets = detect_onsets(y, sr, sensitivity)
        onset_times = librosa.frames_to_time(onsets, sr=sr)
        print(f"  Detected onsets: {len(onsets)}")
        iois = compute_iois(onset_times)
        if len(iois) > 0:
            mean_ioi = np.mean(iois)
            print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
            lattice_base, lattice_coherence = improved_rhythm_lattice(iois)
            print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
        else:
            print("  No IOIs detected.")
        cqt = compute_cqt(y, sr)
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        invariance = cqt_shift_invariance(cqt, sr)
        print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")
        return
    
    for wav in sorted(wav_files):
        print(f"Analysis for {wav}:")
        analyze_file(wav)

if __name__ == "__main__":
    main()