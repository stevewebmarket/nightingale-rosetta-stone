# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

def detect_centroid_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def compute_onsets_and_ioi(y, sr):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    if len(onsets) < 2:
        return onsets, 0.0, 0.0
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    coherence = 1 - np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    return onsets, mean_ioi, coherence

def compute_rhythm_lattice(iois, base_resolution=0.001):
    if len(iois) == 0:
        return base_resolution, 0.0
    min_ioi = np.min(iois) if len(iois) > 0 else base_resolution
    lattice_base = max(min_ioi / 10, base_resolution)
    quantized = np.round(iois / lattice_base) * lattice_base
    residuals = np.abs(iois - quantized)
    coherence = 1 - np.mean(residuals) / np.mean(iois) if np.mean(iois) > 0 else 0
    return lattice_base, coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48)
    return np.abs(cqt)

def cqt_shift_invariance(cqt, shifts=[1, 2, 3]):
    orig = np.mean(cqt, axis=1)
    invariance = 0
    for shift in shifts:
        shifted = np.roll(cqt, shift, axis=0)
        shifted_mean = np.mean(shifted, axis=1)
        corr = np.corrcoef(orig, shifted_mean)[0, 1]
        invariance += corr / len(shifts)
    return invariance

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    centroid_type = detect_centroid_type(y, sr)
    onsets, mean_ioi, rhythm_coherence = compute_onsets_and_ioi(y, sr)
    iois = np.diff(onsets) if len(onsets) > 1 else np.array([])
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois)
    cqt = compute_cqt(y, sr)
    invariance = cqt_shift_invariance(cqt)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_type} sound.")
    print(f"  Detected onsets: {len(onsets)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)