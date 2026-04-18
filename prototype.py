# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Adaptivity, Coherence Metrics, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from math import gcd
from functools import reduce

# Available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Fallback synthetic signals if no files
if not wav_files:
    # Synthetic test signals
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y_sine = np.sin(2 * np.pi * 440 * t)  # Sine wave
    y_noise = np.random.randn(len(t))  # Noise
    y_chirp = librosa.chirp(fmin=100, fmax=10000, sr=sr, duration=5)
    analyses = [('sine', y_sine), ('noise', y_noise), ('chirp', y_chirp)]
else:
    analyses = [(file, None) for file in wav_files]

def classify_sound_type(centroid_mean):
    if centroid_mean > 5000:
        return "high-centroid", {"pre_max": 0.03, "post_max": 0.03, "delta": 0.05, "wait": 0.03}  # High sensitivity
    elif centroid_mean > 2000:
        return "mid-centroid", {"pre_max": 0.05, "post_max": 0.05, "delta": 0.1, "wait": 0.05}   # Mid sensitivity
    else:
        return "low-centroid", {"pre_max": 0.1, "post_max": 0.1, "delta": 0.2, "wait": 0.1}     # Low sensitivity

def compute_rhythm_lattice(iois, sr, onset_times):
    if len(iois) < 2:
        return 0.01, 0.0  # Default
    # Improved: Use quantized IOIs with adaptive binning for better lattice
    hist, bin_edges = np.histogram(iois, bins=50)
    mode_ioi = bin_edges[np.argmax(hist)]  # Approximate mode
    # GCD-based base, but refined with multiples
    quantized_iois = np.round(iois / mode_ioi) * mode_ioi
    gcd_base = reduce(gcd, [int(i * sr) for i in quantized_iois if i > 0]) / sr
    base = max(gcd_base, 0.001)  # Avoid zero
    # Lattice coherence: fraction of onsets fitting the lattice, with entropy for distribution
    lattice_positions = np.round(onset_times / base) * base
    residuals = np.abs(onset_times - lattice_positions)
    fit_fraction = np.mean(residuals < (base / 4))  # Tolerance
    ioi_entropy = entropy(hist + 1e-10) / np.log(len(hist))  # Normalized entropy
    coherence = fit_fraction * (1 - ioi_entropy)
    return base, coherence

def compute_cqt_invariance(cqt, shifts=[1, 2, 4]):
    # Improved: Measure invariance to small time shifts with correlation averaging
    invariance_scores = []
    for shift in shifts:
        if shift >= cqt.shape[1]:
            continue
        shifted = np.roll(cqt, shift, axis=1)
        corr_matrix = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        invariance_scores.append(corr_matrix)
    # Enhance for broad sounds: Weight by spectral consistency
    spec_consistency = 1 - np.std(np.mean(np.abs(cqt), axis=1)) / np.mean(np.abs(cqt))
    return np.mean(invariance_scores) * spec_consistency if invariance_scores else 0.0

for name, y_input in analyses:
    if y_input is None:
        y, sr = librosa.load(name, sr=22050)
    else:
        y = y_input
        sr = 22050
    print(f"Analysis for {name}{'.wav' if '.' not in name else ''}:")
    
    # Spectral centroid for classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    sound_type, onset_params = classify_sound_type(centroid_mean)
    print(f"  Detected {sound_type} sound.")
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    # Onset detection with adaptive params
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    # IOIs and rhythm coherence
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s", end="")
    # Improved coherence: CV with regularization for broad sounds
    cv = np.std(iois) / (mean_ioi + 1e-5)
    rhythm_coherence = max(0, 1 - cv) * (1 - np.exp(-len(iois)/100))  # Penalize few onsets
    print(f", rhythm coherence: {rhythm_coherence:.2f}")
    
    # Rhythm lattice
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois, sr, onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with improved params for invariance and broad handling
    hop_length = 512 if "high" in sound_type else 1024 if "mid" in sound_type else 2048  # Adaptive hop
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # CQT shift invariance
    invariance = compute_cqt_invariance(np.abs(cqt))
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")