# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice, Enhanced Coherence, CQT Invariance Improvements
# =============================================================================

import librosa
import numpy as np

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f'Analysis for {file}:')
    y, sr = librosa.load(file, sr=22050)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 2000:
        centroid_type = 'low'
    elif mean_centroid < 5000:
        centroid_type = 'mid'
    else:
        centroid_type = 'high'
    print(f'  Detected {centroid_type}-centroid sound.')

    # Onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onsets = len(onset_frames)
    print(f'  Detected onsets: {onsets}')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    iois = np.diff(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 15 * np.exp(-cv)  # Enhanced coherence metric
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')

    # Adaptive rhythm lattice
    threshold = 0.8
    best_base = 0
    best_coh = 0
    if len(iois) > 0:
        for div in range(1, 33):
            candidate_base = mean_ioi / div
            if candidate_base < 0.001:
                break
            ratios = iois / candidate_base
            rounded = np.round(ratios)
            rel_error = np.abs(ratios - rounded) / (rounded + 1e-10)
            num_fit = np.sum(rel_error < 0.05)
            coh = num_fit / len(iois)
            if coh >= threshold and candidate_base > best_base:
                best_base = candidate_base
                best_coh = coh
    if best_base == 0:
        best_base = 0.023 if len(iois) == 0 else mean_ioi / 4
        best_coh = 1.0 if len(iois) == 0 else 0.5
    print(f'  Rhythm lattice base: {best_base:.3f} s')
    print(f'  lattice coherence: {best_coh:.2f}')

    # CQT with adaptive hop_length for broad sound handling and improved invariance
    hop_length = 256 if centroid_type == 'high' else 512 if centroid_type == 'mid' else 1024
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48))
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')

    # Enhanced CQT shift invariance metric (per-bin normalized correlation)
    if cqt.shape[1] > 1:
        cqt_norm = (cqt - np.min(cqt, axis=1, keepdims=True)) / (np.max(cqt, axis=1, keepdims=True) - np.min(cqt, axis=1, keepdims=True) + 1e-10)
        shifted = np.roll(cqt_norm, 1, axis=1)
        corrs = [np.corrcoef(cqt_norm[i], shifted[i])[0, 1] for i in range(cqt.shape[0])]
        invariance = np.mean([c for c in corrs if not np.isnan(c)])
    else:
        invariance = 0
    print(f'  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)')