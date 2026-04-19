# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import variation
from math import gcd
from functools import reduce

def classify_sound_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        return "low-centroid"
    elif mean_centroid > 4000:
        return "high-centroid"
    else:
        return "mid-centroid"

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    # Rhythm coherence: 1 / (1 + coefficient of variation)
    cv = variation(iois)
    coherence = 1 / (1 + cv) if not np.isnan(cv) else 0.0
    return len(onset_times), mean_ioi, coherence

def compute_adaptive_lattice_base(iois):
    if len(iois) == 0:
        return 0.001
    # Approximate IOIs to milliseconds for GCD computation
    iois_ms = (iois * 1000).astype(int)
    # Find GCD of IOIs, fall back to min IOI / 10 if GCD is 0
    gcd_val = reduce(gcd, iois_ms) if min(iois_ms) > 0 else 1
    base = max(gcd_val / 1000.0, min(iois) / 10)
    return base

def compute_lattice_coherence(onset_times, base):
    if len(onset_times) < 2 or base <= 0:
        return 0.0
    # Quantize onsets to lattice
    quantized = np.round(onset_times / base) * base
    errors = np.abs(onset_times - quantized)
    max_error = base / 2.0
    # Coherence: average fit (1 - normalized error)
    norm_errors = errors / max_error
    coherence = np.mean(1 - norm_errors)
    return coherence

def compute_cqt_with_invariance(y, sr, n_bins=192, bins_per_octave=24):
    # Use higher bins_per_octave for finer resolution and better shift invariance
    cqt = np.abs(librosa.cqt(y=y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave))
    # Normalize for better invariance
    cqt_norm = librosa.util.normalize(cqt, axis=0)
    # Compute shift invariance: correlation with octave-shifted version
    shift_bins = bins_per_octave  # One octave shift
    if shift_bins < n_bins:
        shifted = np.roll(cqt_norm, shift_bins, axis=0)
        # Correlation coefficient
        corr = np.corrcoef(cqt_norm.flatten(), shifted.flatten())[0, 1]
    else:
        corr = 0.0
    return cqt_norm, corr

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    centroid_class = classify_sound_centroid(y, sr)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(onset_times)
    iois = np.diff(onset_times) if len(onset_times) > 1 else np.array([])
    lattice_base = compute_adaptive_lattice_base(iois)
    lattice_coherence = compute_lattice_coherence(onset_times, lattice_base)
    cqt, invariance_metric = compute_cqt_with_invariance(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_class} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

def generate_synthetic_signal(sr=22050, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)  # A4 tone
    return y

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        print("No WAV files available. Using synthetic test signal.")
        y = generate_synthetic_signal()
        # Save or analyze synthetic, but for demo, analyze as 'synthetic.wav'
        analyze_audio('synthetic.wav')  # Placeholder, assumes y is loaded
    else:
        for file in files:
            analyze_audio(file)