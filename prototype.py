# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f'Analysis for {file}:')
    y, sr = librosa.load(file, sr=22050)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        print('  Detected high-centroid sound.')
    else:
        print('  Detected mid-centroid sound.')
    # Onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    print(f'  Detected onsets: {len(onset_times)}')
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        coherence = 1 / (1 + np.std(iois) / mean_ioi)
        print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coherence:.2f}')
        # Improved rhythm lattice: use gcd of ioi in samples, fallback to tempo-based if gcd too small
        onset_samples = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='samples')
        iois_samples = np.diff(onset_samples)
        if len(iois_samples) > 1:
            lattice_samples = reduce(gcd, iois_samples.astype(int))
            lattice_base = lattice_samples / sr
            if lattice_base < 0.001:  # too small, fallback to tempo subdivision
                tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
                lattice_base = 60 / tempo / 4  # quarter beat
        else:
            lattice_base = mean_ioi
        print(f'  Rhythm lattice base: {lattice_base:.3f} s')
        # Improved lattice coherence: fraction of onsets within tolerance of lattice multiples
        tol = 0.05 * lattice_base
        num_aligned = 0
        lattice_points = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
        for ot in onset_times:
            if np.min(np.abs(ot - lattice_points)) < tol:
                num_aligned += 1
        coherence = num_aligned / len(onset_times)
        print(f'  lattice coherence: {coherence:.2f}')
    # CQT with improvements for invariance and broad handling
    hop_length = 256  # smaller for better temporal resolution
    fmin = librosa.note_to_hz('C1')
    if centroid > 5000:
        fmin = librosa.note_to_hz('C3')  # higher for high-centroid sounds
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=384, bins_per_octave=48))
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')
    # Improved shift invariance metric: average cosine similarity between consecutive frames
    cqt_norm = cqt / (np.linalg.norm(cqt, axis=0, keepdims=True) + 1e-8)
    sims = []
    for i in range(cqt.shape[1] - 1):
        sim = np.dot(cqt_norm[:, i], cqt_norm[:, i + 1])
        sims.append(sim)
    invariance = np.mean(sims) if sims else 0.0
    print(f'  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)')
    print()