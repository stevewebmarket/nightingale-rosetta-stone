# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Optimized rhythm lattice, fixed tempo call, added high centroid, improved CQT params
# =============================================================================

import librosa
import numpy as np
from scipy.stats import pearsonr

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    if centroid < 1500:
        print("  Detected low-centroid sound.")
        sensitivity = 'low'
        delta = 0.1
        hop_length = 512
        filter_scale = 1.0
    elif centroid < 4000:
        print("  Detected mid-centroid sound.")
        sensitivity = 'mid'
        delta = 0.07
        hop_length = 512
        filter_scale = 1.5
    else:
        print("  Detected high-centroid sound.")
        sensitivity = 'high'
        delta = 0.04
        hop_length = 256
        filter_scale = 2.0
    print(f"  Using {sensitivity}-sensitivity onset params.")
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, delta=delta)
    print(f"  Detected onsets: {len(onsets)}")
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    from librosa.feature.rhythm import tempo
    tempo_bpm = tempo(onset_envelope=o_env, sr=sr, hop_length=hop_length)[0]
    lattice_base = 60 / tempo_bpm if tempo_bpm > 0 else 1
    duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)
    candidate_bases = np.linspace(max(0.1, lattice_base/2), lattice_base*2, 50)
    best_coherence = 0
    best_base = lattice_base
    for cb in candidate_bases:
        if cb == 0: continue
        lattice_points = np.arange(0, duration + cb, cb)
        coh = 0
        for ot in onset_times:
            dists = np.abs(ot - lattice_points)
            min_dist = np.min(dists)
            coh += 1 - min_dist / (cb / 2) if min_dist < cb / 2 else 0
        coh /= len(onset_times) if len(onset_times) > 0 else 1
        if coh > best_coherence:
            best_coherence = coh
            best_base = cb
    print(f"  Rhythm lattice base: {best_base:.3f} s")
    print(f"  lattice coherence: {best_coherence:.2f}")
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48, filter_scale=filter_scale)
    cqt_mag = np.abs(cqt)
    print(f"  CQT shape: {cqt_mag.shape}, n_bins: {cqt_mag.shape[0]}")
    invariance = 0
    num_bins = cqt_mag.shape[0]
    for i in range(num_bins):
        if cqt_mag.shape[1] > 1:
            corr = pearsonr(cqt_mag[i, :-1], cqt_mag[i, 1:])[0]
            invariance += max(corr, 0)
    invariance /= num_bins
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")