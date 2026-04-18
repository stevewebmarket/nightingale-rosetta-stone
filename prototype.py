# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Dynamic Rhythm Lattice + Multi-Centroid Onset + Smoothed CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce
from scipy.ndimage import gaussian_filter

def approximate_gcd(times):
    if len(times) == 0:
        return 0.010
    times_ms = np.array(times) * 1000
    times_int = np.round(times_ms).astype(int)
    if np.all(times_int == 0):
        return 0.010
    gcd_ms = reduce(gcd, times_int)
    return max(gcd_ms / 1000.0, 0.001)  # Avoid zero or too small

def analyze(file):
    y, sr = librosa.load(file, sr=22050)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_cent = np.mean(centroid)

    if mean_cent < 1500:
        cent_type = 'low'
        onset_desc = 'low-sensitivity'
        delta = 0.1
        pre_max = 0.05
        hop_length = 512
    elif mean_cent > 4000:
        cent_type = 'high'
        onset_desc = 'high-sensitivity'
        delta = 0.02
        pre_max = 0.01
        hop_length = 256  # Smaller for better resolution in transients
    else:
        cent_type = 'mid'
        onset_desc = 'standard-sensitivity'
        delta = 0.05
        pre_max = 0.03
        hop_length = 512

    print(f"  Detected {cent_type}-centroid sound{' (e.g., bass-heavy)' if cent_type == 'low' else ' (e.g., orchestral or rock)' if cent_type == 'mid' else ' (e.g., birdsong or high-frequency)' }.")
    print(f"  Using {onset_desc} onset params.")

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, delta=delta, pre_max=pre_max, backtrack=True, hop_length=hop_length)
    print(f"  Detected onsets: {len(onset_frames)}")

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        rhythm_coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")

        lattice_base = approximate_gcd(iois)
        multiples = iois / lattice_base
        closeness = np.abs(multiples - np.round(multiples))
        lattice_coherence = np.mean(closeness < 0.1)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    else:
        print("  Insufficient onsets for rhythm analysis.")

    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=216, bins_per_octave=36)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

    cqt_abs = np.abs(cqt)
    cqt_sm = gaussian_filter(cqt_abs, sigma=[0, 1])  # Smooth along time axis for better invariance
    if cqt_sm.shape[1] > 1:
        diff = np.abs(cqt_sm[:, 1:] - cqt_sm[:, :-1])
        metric = np.mean(diff) / (np.mean(cqt_sm) + 1e-6)
        print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    available_files = [f for f in ['birdsong.wav', 'orchestra.wav', 'rock.wav'] if os.path.exists(f)]
    if not available_files:
        # Fallback to synthetic test signals
        print("No WAV files found. Generating synthetic test signal.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr, endpoint=False)
        y = np.sin(440 * 2 * np.pi * t) + 0.5 * np.sin(880 * 2 * np.pi * t)
        analyze_synthetic = lambda: analyze(None)  # Placeholder; adapt if needed
        # For simplicity, save as temp and load, but to keep pure, simulate
        # Here, we'd set file='synthetic' and pass y, sr directly, but to match structure, skip details
        print("Synthetic analysis not fully implemented in this prototype.")
    else:
        for file in available_files:
            print(f"Analysis for {file}:")
            analyze(file)