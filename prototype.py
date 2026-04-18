# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice via Autocorrelation + Adaptive CQT Params
# =============================================================================

import librosa
import numpy as np
from scipy.signal import find_peaks

def cqt_shift_invariance(cqt):
    mag = np.abs(cqt)
    norms = np.linalg.norm(mag, axis=0)
    norms[norms == 0] = 1
    mag_norm = mag / norms
    sims = []
    for i in range(1, mag_norm.shape[1]):
        dot = np.dot(mag_norm[:, i-1], mag_norm[:, i])
        sims.append(dot)
    if sims:
        return np.mean(sims)
    else:
        return 0

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_cent = np.mean(cent)
    is_high_centroid = mean_cent > 5000
    print(f"  Detected {'high' if is_high_centroid else 'mid'}-centroid sound.")
    onset_params = {
        'backtrack': is_high_centroid,
        'delta': 0.02 if is_high_centroid else 0.07,
        'wait': 1 if is_high_centroid else 4
    }
    print(f"  Using {'high' if is_high_centroid else 'mid'}-sensitivity onset params.")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, **onset_params)
    print(f"  Detected onsets: {len(onset_frames)}")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv)
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    ac = np.correlate(oenv, oenv, mode='full')[len(oenv)-1:]
    ac = ac / ac[0] if ac[0] != 0 else ac
    peaks, _ = find_peaks(ac, height=0.1, distance=10)
    if len(peaks) > 0:
        dominant_period_frames = peaks[0]
        lattice_base = dominant_period_frames * (512 / sr)
        lattice_coherence = ac[peaks[0]]
    else:
        lattice_base = 0
        lattice_coherence = 0
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    fmin = 500 if is_high_centroid else None
    cqt = librosa.cqt(y=y, sr=sr, hop_length=512, fmin=fmin, n_bins=384, bins_per_octave=48)
    print(f"  CQT shape: ({cqt.shape[0]}, {cqt.shape[1]}), n_bins: {cqt.shape[0]}")
    invariance_metric = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")