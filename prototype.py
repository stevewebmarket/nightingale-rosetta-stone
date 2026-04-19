# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice and Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    y, sr = librosa.load(file, sr=22050)

    # Spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 500:
        cent_type = 'low'
    elif mean_centroid < 2000:
        cent_type = 'mid'
    else:
        cent_type = 'high'

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onsets = len(onset_times)

    # Inter-onset intervals (IOI) and rhythm coherence
    if onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        # Improved coherence: coefficient of variation inverse, normalized
        cv = np.std(iois) / mean_ioi
        coh = 1 / (1 + cv)
    else:
        mean_ioi = 0
        coh = 0

    # Adaptive rhythm lattice base (improved: based on mean IOI for better fit across sound types)
    if mean_ioi > 0:
        lattice_base = mean_ioi / 256  # Finer grid for broad handling, power-of-2 division
    else:
        lattice_base = 0.001

    # Lattice coherence (improved: mean deviation normalized by base, with clamping)
    if onsets > 0:
        lattice_times = np.arange(0, len(y)/sr, lattice_base)
        diffs = []
        for ot in onset_times:
            closest = np.argmin(np.abs(lattice_times - ot))
            diffs.append(np.abs(lattice_times[closest] - ot))
        mean_diff = np.mean(diffs)
        lat_coh = max(0, 1 - (mean_diff / lattice_base))  # Clamp to [0,1] for coherence
    else:
        lat_coh = 0

    # CQT computation (improved: hybrid CQT for better invariance and broad sound handling)
    cqt = librosa.hybrid_cqt(y, sr=sr, n_bins=384)
    cqt_mag = np.abs(cqt)
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]

    # CQT shift invariance metric (improved: average correlation over multiple small shifts for better metric applicability)
    corrs = []
    max_shift = min(5, cqt_shape[1] - 1)
    for shift in range(1, max_shift + 1):
        corr = np.corrcoef(cqt_mag[:, :-shift].flatten(), cqt_mag[:, shift:].flatten())[0, 1]
        corrs.append(corr)
    invariance = np.mean(corrs) if corrs else 0.0

    # Output
    print(f"Analysis for {file}:")
    print(f"  Detected {cent_type}-centroid sound.")
    print(f"  Detected onsets: {onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coh:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lat_coh:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")