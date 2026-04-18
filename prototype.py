# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_cent = np.mean(centroid)
    
    if mean_cent < 2000:
        sound_type = 'low-centroid sound.'
        sensitivity = 'low'
        onset_params = {'backtrack': True, 'delta': 0.1}
    elif mean_cent < 6000:
        sound_type = 'mid-centroid sound.'
        sensitivity = 'mid'
        onset_params = {'backtrack': False, 'delta': 0.05}
    else:
        sound_type = 'high-centroid sound.'
        sensitivity = 'high'
        onset_params = {'backtrack': False, 'delta': 0.02}
    
    print(f"  Detected {sound_type}")
    print(f"  Using {sensitivity}-sensitivity onset params.")
    
    # Onset detection
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
    print(f"  Detected onsets: {len(onset_times)}")
    
    # Rhythm metrics
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv)
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Rhythm lattice base (improved to better capture subdivisions)
    if mean_ioi > 0:
        base = mean_ioi / 20
    else:
        base = 0.01
    print(f"  Rhythm lattice base: {base:.3f} s")
    
    # Improved lattice coherence calculation
    if len(onset_times) > 1:
        quantized = np.round(onset_times / base) * base
        residuals = np.abs(onset_times - quantized) / base
        mean_res = np.mean(residuals)
        lattice_coherence = 1 - np.clip(mean_res / 0.5, 0, 1)
    else:
        lattice_coherence = 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with increased bins for better shift invariance
    n_bins = 672
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=n_bins // 7)
    cqt_mag = np.abs(cqt)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    # CQT shift invariance metric
    shifted = np.roll(cqt_mag, 1, axis=0)
    diff = np.abs(shifted - cqt_mag)
    metric = np.mean(diff) / (np.mean(cqt_mag) + 1e-6)  # Avoid division by zero
    print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")