# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return "high"
    elif mean_centroid > 2000:
        return "mid"
    else:
        return "low"

def adjust_onset_params(centroid_type):
    if centroid_type == "high":
        return {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.05}
    elif centroid_type == "mid":
        return {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.1}
    else:
        return {'backtrack': False, 'pre_max': 0.1, 'post_max': 0.1, 'delta': 0.2}

def detect_onsets(y, sr, params):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, **params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times

def compute_iois(onset_times):
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    return mean_ioi, iois

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    std_ioi = np.std(iois)
    mean_ioi = np.mean(iois)
    return 1 / (1 + std_ioi / mean_ioi) if mean_ioi > 0 else 0.0

def estimate_rhythm_lattice(iois):
    if len(iois) == 0:
        return 0.0
    hist, bin_edges = np.histogram(iois, bins=50)
    base = bin_edges[np.argmax(hist)]
    return base

def lattice_coherence(onset_times, base):
    if base == 0 or len(onset_times) == 0:
        return 0.0
    lattice = np.arange(onset_times[0], onset_times[-1] + base, base)
    matches = 0
    for onset in onset_times:
        if np.min(np.abs(lattice - onset)) < base * 0.05:
            matches += 1
    return matches / len(onset_times)

def compute_cqt(y, sr):
    n_octaves = 8
    bins_per_octave = 84
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=n_octaves * bins_per_octave, bins_per_octave=bins_per_octave, filter_scale=1.0, fmin=librosa.note_to_hz('C1'))
    cqt_mag = np.abs(cqt)
    return cqt_mag

def cqt_shift_invariance(cqt_mag, shift=1):
    if cqt_mag.shape[1] < shift + 1:
        return 0.0
    shifted = cqt_mag[:, shift:]
    original = cqt_mag[:, :-shift]
    min_len = min(shifted.shape[1], original.shape[1])
    shifted = shifted[:, :min_len]
    original = original[:, :min_len]
    corr = np.mean(np.abs(np.corrcoef(shifted.flatten(), original.flatten())))
    return corr

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    centroid_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_type}-centroid sound.")
    params = adjust_onset_params(centroid_type)
    print(f"  Using {centroid_type}-sensitivity onset params.")
    onset_times = detect_onsets(y, sr, params)
    print(f"  Detected onsets: {len(onset_times)}")
    mean_ioi, iois = compute_iois(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
    lattice_base = estimate_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence(onset_times, lattice_base):.2f}")
    cqt_mag = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt_mag.shape}, n_bins: {cqt_mag.shape[0]}")
    invariance = cqt_shift_invariance(cqt_mag)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)

if __name__ == "__main__":
    main()