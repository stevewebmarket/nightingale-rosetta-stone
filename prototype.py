# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

def detect_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        return "low-centroid sound"
    elif mean_centroid < 4000:
        return "mid-centroid sound"
    else:
        return "high-centroid sound"

def compute_iois(onset_times):
    return np.diff(onset_times)

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, _ = np.histogram(iois, bins=20)
    return 1 - entropy(hist) / np.log(len(hist))

def find_rhythm_lattice(iois, base_resolution=0.01):
    quantized = np.round(iois / base_resolution).astype(int)
    if len(quantized) == 0:
        return base_resolution
    base_units = reduce(gcd, quantized)
    return base_units * base_resolution

def lattice_coherence(onset_times, lattice_base):
    if lattice_base == 0:
        return 0.0
    grid = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
    hits = 0
    for ot in onset_times:
        if np.min(np.abs(grid - ot)) < lattice_base / 2:
            hits += 1
    return hits / len(onset_times)

def cqt_shift_invariance(cqt, shifts=[1, 2, 4]):
    orig = np.abs(cqt)
    invariance = 0.0
    for shift in shifts:
        shifted = np.roll(orig, shift, axis=0)
        diff = np.mean(np.abs(orig - shifted))
        invariance += 1 / (1 + diff)
    return invariance / len(shifts)

def analyze_audio(filename, sr=22050):
    y, sr = librosa.load(filename, sr=sr)
    print(f"Analysis for {filename}:")
    
    sound_type = detect_sound_type(y, sr)
    print(f"  Detected {sound_type}.")
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_times)}")
    
    iois = compute_iois(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois) * 100:.2f}")
        
        lattice_base = find_rhythm_lattice(iois, base_resolution=0.005)  # Finer resolution for better lattice
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        
        coh = lattice_coherence(onset_times, lattice_base)
        print(f"  lattice coherence: {coh:.2f}")
    else:
        print("  No IOIs detected.")
    
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.5)  # Adjusted for better invariance
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print()

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)

if __name__ == "__main__":
    main()