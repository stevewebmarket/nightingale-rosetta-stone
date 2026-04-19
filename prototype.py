# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Optimized Rhythm Lattice, Adaptive CQT, Improved Coherence
# =============================================================================

import librosa
import numpy as np
from librosa.feature.rhythm import tempo

def compute_octave_invariance(cqt, shift):
    if cqt.shape[0] < shift + 1:
        return 0.0
    min_bins = cqt.shape[0] - shift
    corrs = []
    for t in range(cqt.shape[1]):
        orig = cqt[:min_bins, t]
        shifted = cqt[shift:shift + min_bins, t]
        if np.all(orig == 0) or np.all(shifted == 0):
            continue
        corr = np.corrcoef(orig, shifted)[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    return np.mean(corrs) if corrs else 0.0

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)

    # Spectral centroid for type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 2000:
        sound_type = "low-centroid sound"
        fmin = librosa.note_to_hz('A0')
    elif mean_centroid > 6000:
        sound_type = "high-centroid sound"
        fmin = librosa.note_to_hz('C3')
    else:
        sound_type = "mid-centroid sound"
        fmin = librosa.note_to_hz('C1')
    print(f"  Detected {sound_type}.")

    # Onsets
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr, units='time', backtrack=True)
    num_onsets = len(onset_times)
    print(f"  Detected onsets: {num_onsets}")

    if num_onsets < 2:
        return

    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 20 / (cv + 1)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")

    # Tempo
    temp = tempo(onset_envelope=onset_strength, sr=sr)[0]
    if temp <= 0:
        temp = 60 / mean_ioi
    beat_duration = 60 / temp

    # Optimized rhythm lattice
    best_coherence = 0
    best_lattice_base = 0
    duration = librosa.get_duration(y=y, sr=sr)
    for div in range(8, 33):
        lattice_base = beat_duration / div
        lattice_points = np.arange(0, duration + lattice_base, lattice_base)
        num_fits = 0
        tolerance = lattice_base * 0.3
        for ot in onset_times:
            dists = np.abs(ot - lattice_points)
            if np.min(dists) < tolerance:
                num_fits += 1
        coherence = num_fits / num_onsets if num_onsets > 0 else 0
        if coherence > best_coherence:
            best_coherence = coherence
            best_lattice_base = lattice_base
    print(f"  Rhythm lattice base: {best_lattice_base:.3f} s")
    print(f"  lattice coherence: {best_coherence:.2f}")

    # Adaptive CQT
    cqt = np.abs(librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=384, bins_per_octave=48))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

    # CQT shift invariance metric (octave invariance)
    invariance = compute_octave_invariance(cqt, 48)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ["birdsong.wav", "orchestra.wav", "rock.wav"]
    for file in files:
        print(f"Analysis for {file}:")
        analyze_audio(file)