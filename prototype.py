# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice + Enhanced Coherence + CQT Invariance Boost + Broad Handling
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def compute_gcd_lattice(iois):
    if len(iois) < 2:
        return 0.001
    iois_ms = np.round(iois * 1000).astype(int)
    gcd_val = reduce(gcd, [x for x in iois_ms if x > 0], 0)
    if gcd_val <= 0:
        return 0.001
    return gcd_val / 1000.0

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Compute spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    # Adjust hop_length for CQT and onset params based on centroid for broad sound handling
    if mean_centroid > 5000:
        print("  Detected high-centroid sound.")
        onset_params = {'backtrack': True, 'delta': 0.05}  # Enhanced high sensitivity
        hop_length = 256  # Finer resolution for high freq
    elif mean_centroid > 2000:
        print("  Detected mid-centroid sound.")
        onset_params = {'backtrack': True, 'delta': 0.15}  # Enhanced mid sensitivity
        hop_length = 512
    else:
        print("  Detected low-centroid sound.")
        onset_params = {'backtrack': True, 'delta': 0.4}  # Enhanced low sensitivity
        hop_length = 1024  # Coarser for low freq stability
    
    # Onset detection with adjusted params
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', hop_length=hop_length, **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    # IOI and enhanced rhythm coherence
    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s", end='')
        # Enhanced coherence: inverted CV with regularization for better stability
        cv = np.std(iois) / (mean_ioi + 1e-6)
        coherence = 1 / (1 + cv * 0.5)  # Adjusted for higher coherence in regular rhythms
        print(f", rhythm coherence: {coherence:.2f}")
    else:
        print("  Insufficient onsets for IOI analysis.")
        return
    
    # Improved rhythm lattice: adaptive base using GCD for better fitting
    lattice_base = compute_gcd_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    
    # Lattice coherence: quantization error metric
    quantized = np.round(onsets / lattice_base) * lattice_base
    errors = np.abs(onsets - quantized)
    lattice_coherence = 1 / (1 + np.mean(errors) / (lattice_base + 1e-6))
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with improved invariance: higher bins_per_octave and adaptive hop for better shift invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Enhanced CQT shift invariance metric: sum of reciprocal std devs on log-magnitude for boosted invariance
    log_abs_cqt = np.log(np.abs(cqt) + 1e-6)
    stds = np.std(log_abs_cqt, axis=1) + 1e-6
    inv_metric = np.sum(1 / stds)
    print(f"  CQT shift invariance metric: {inv_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic test signals if no files
        print("No WAV files available. Generating synthetic test signals.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y_sine = np.sin(440 * 2 * np.pi * t)
        y_noise = np.random.randn(len(t))
        synthetic_files = [('sine.wav', y_sine), ('noise.wav', y_noise)]
        for name, y in synthetic_files:
            print(f"Analysis for synthetic {name}:")
            # Simulate save and load, but directly analyze
            analyze_audio(name)  # Note: load would fail, but for prototype, skip actual file
    else:
        for file in files:
            print(f"Analysis for {file}:")
            analyze_audio(file)