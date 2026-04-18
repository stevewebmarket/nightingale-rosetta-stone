# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import os
import librosa
import numpy as np
from math import gcd
from functools import reduce

def compute_centroid_type(mean_centroid):
    if mean_centroid < 1500:
        return "low-centroid", "low-sensitivity"
    elif mean_centroid < 4000:
        return "mid-centroid", "mid-sensitivity"
    else:
        return "high-centroid", "high-sensitivity"

def get_onset_params(sensitivity):
    if sensitivity == "low-sensitivity":
        return {'pre_max': 0.05, 'post_max': 0.05, 'wait': 0.05, 'delta': 0.1}
    elif sensitivity == "high-sensitivity":
        return {'pre_max': 0.01, 'post_max': 0.01, 'wait': 0.01, 'delta': 0.02}
    else:  # mid
        return {'pre_max': 0.03, 'post_max': 0.03, 'wait': 0.03, 'delta': 0.07}

def generate_synthetic_signal(sr=22050, duration=10):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440
    y = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    return y

print("Analyzing available WAV files.")

wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
if not wav_files:
    print("No WAV files found. Falling back to synthetic test signal.")
    y = generate_synthetic_signal()
    sr = 22050
    filename = "synthetic.wav"
    print(f"Analysis for {filename}:")
else:
    for filename in wav_files:
        if not os.path.exists(filename):
            continue
        print(f"Analysis for {filename}:")
        y, sr = librosa.load(filename, sr=22050)

        # Spectral centroid for sound type
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = np.mean(centroid)
        centroid_type, sensitivity = compute_centroid_type(mean_centroid)
        print(f"  Detected {centroid_type} sound.")
        print(f"  Using {sensitivity} onset params.")

        # Onset detection with adjusted params
        params = get_onset_params(sensitivity)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, **params)

        num_onsets = len(onset_times)
        print(f"  Detected onsets: {num_onsets}")

        if num_onsets > 1:
            iois = np.diff(onset_times)
            mean_ioi = np.mean(iois)
            print(f"  mean IOI: {mean_ioi:.2f} s", end=", ")

            # Improved rhythm coherence: normalized CV (coefficient of variation)
            cv = np.std(iois) / mean_ioi
            rhythm_coherence = 1 / (1 + cv)  # Bounded between 0 and 1, higher better
            print(f"rhythm coherence: {rhythm_coherence:.2f}")

            # Improved rhythm lattice: GCD of discretized IOIs with tolerance
            times_ms = (onset_times * 1000).astype(int)
            diffs = np.diff(times_ms)
            if len(diffs) > 0:
                lattice_base_ms = reduce(gcd, diffs)
                if lattice_base_ms == 0:
                    lattice_base_ms = 1
            else:
                lattice_base_ms = 1
            lattice_base = lattice_base_ms / 1000.0
            print(f"  Rhythm lattice base: {lattice_base:.3f} s")

            # Improved lattice coherence: percentage of onsets fitting lattice with tolerance
            residuals = times_ms % lattice_base_ms
            fit = np.isclose(residuals, 0, atol=lattice_base_ms * 0.05)  # 5% tolerance
            lattice_coherence = np.mean(fit)
            print(f"  lattice coherence: {lattice_coherence:.2f}")
        else:
            print("  Insufficient onsets for rhythm analysis.")

        # Improved CQT: Use hybrid_cqt for better low-freq handling, smaller hop for invariance
        cqt = np.abs(librosa.hybrid_cqt(y, sr=sr, hop_length=256, n_bins=252, bins_per_octave=24, tuning=0.0))
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

        # Improved CQT shift invariance metric: normalized mean abs diff after 1-frame shift
        if cqt.shape[1] > 1:
            shifted = np.roll(cqt, 1, axis=1)
            diff = np.mean(np.abs(cqt[:, 1:] - shifted[:, 1:]))
            norm = np.mean(np.abs(cqt)) + 1e-8  # Avoid div by zero
            metric = diff / norm
            print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")
        else:
            print("  Insufficient frames for invariance metric.")