# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Refined Rhythm Lattice and CQT Invariance
# =============================================================================

import librosa
import numpy as np

def analyze_audio(filename):
    y, sr = librosa.load(filename, sr=22050)
    # Spectral centroid for classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        sound_type = "high-centroid"
        delta = 0.02  # higher sensitivity, lower delta
        wait = 1
    elif mean_centroid > 2000:
        sound_type = "mid-centroid"
        delta = 0.05
        wait = 2
    else:
        sound_type = "low-centroid"
        delta = 0.1
        wait = 4
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type} sound.")
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    # Onset detection
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=True, delta=delta, wait=wait)
    print(f"  Detected onsets: {len(onsets)}")
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        rhythm_coherence = max(0, 1 - std_ioi / mean_ioi) if mean_ioi > 0 else 0
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    # Improved rhythm lattice
    tempo = librosa.beat.tempo(onset_envelope=o_env, sr=sr)[0]
    lattice_base = 60 / tempo if tempo > 0 else 0.0
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    # Lattice coherence
    if lattice_base > 0 and len(onset_times) > 0:
        phases = (onset_times % lattice_base) / lattice_base
        var_phase = np.var(phases)
        lattice_coherence = max(0, 1 - 12 * var_phase)
    else:
        lattice_coherence = 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    # CQT
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48)
    cqt_abs = np.abs(cqt)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    # Improved CQT invariance: average cosine similarity between consecutive time frames after normalization
    cqt_norm = cqt_abs / (np.linalg.norm(cqt_abs, axis=1, keepdims=True) + 1e-8)
    if cqt_norm.shape[1] > 1:
        corrs = []
        for i in range(cqt_norm.shape[1] - 1):
            vec1 = cqt_norm[:, i]
            vec2 = cqt_norm[:, i+1]
            corr = np.dot(vec1, vec2)  # cosine similarity since normalized
            corrs.append(corr)
        invariance = np.mean(corrs)
    else:
        invariance = 1.0
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
for file in files:
    analyze_audio(file)