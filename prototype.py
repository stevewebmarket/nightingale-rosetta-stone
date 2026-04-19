# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence and CQT Invariance for Broad Sounds
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def compute_lattice_base(iois):
    if len(iois) < 2:
        return 0.001
    iois_ms = np.round(iois * 1000).astype(int)
    base_ms = reduce(gcd, iois_ms)
    if base_ms == 0:
        base_ms = 1
    return base_ms / 1000.0

def compute_lattice_coherence(onset_times, base, duration):
    if base <= 0 or len(onset_times) == 0:
        return 0.0
    lattice = np.arange(0, duration + base, base)
    errors = []
    for ot in onset_times:
        dists = np.abs(lattice - ot)
        min_dist = np.min(dists)
        errors.append(min_dist)
    mean_error = np.mean(errors)
    max_error = base / 2.0
    coherence = max(0.0, 1.0 - (mean_error / max_error))
    return coherence

def compute_cqt_invariance(cqt, bpo=24):
    if cqt.shape[1] == 0:
        return 0.0
    spectrum = np.mean(cqt, axis=1)
    norm = np.linalg.norm(spectrum)
    if norm == 0:
        return 0.0
    spectrum = spectrum / norm
    shifted = np.roll(spectrum, bpo)
    invariance = np.dot(spectrum, shifted)
    return max(0.0, invariance)  # Clip to non-negative for metric

def classify_centroid(mean_cent):
    if mean_cent > 5000:
        return "high-centroid"
    elif mean_cent > 2000:
        return "mid-centroid"
    else:
        return "low-centroid"

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_cent = np.mean(centroid)

    # Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    num_onsets = len(onset_times)

    # IOIs and rhythm coherence
    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = np.exp(-cv)
    else:
        mean_ioi = 0.0
        rhythm_coherence = 0.0

    # Rhythm lattice
    lattice_base = compute_lattice_base(np.diff(onset_times)) if num_onsets > 1 else 0.001
    lattice_coherence = compute_lattice_coherence(onset_times, lattice_base, duration)

    # CQT with adaptive fmin based on centroid
    if mean_cent > 5000:  # High-centroid, e.g., birdsong
        fmin = librosa.note_to_hz('A3')  # Higher fmin for better capture
    else:
        fmin = librosa.note_to_hz('C1')
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, fmin=fmin, n_bins=168, bins_per_octave=24))
    invariance = compute_cqt_invariance(cqt)

    # Output
    print(f"Analysis for {file}:")
    print(f"  Detected {classify_centroid(mean_cent)} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic if no files
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=5)
        np.save('synthetic.npy', y)  # Dummy save, not used
        analyze_audio('synthetic.npy')  # But load would fail, placeholder
    else:
        for file in files:
            analyze_audio(file)

if __name__ == "__main__":
    main()