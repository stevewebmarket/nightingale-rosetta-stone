# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + CQT Invariance Boost + Broad Sound Tuning
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import correlate

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 4000:
        return "high-centroid"
    elif mean_centroid > 2000:
        return "mid-centroid"
    else:
        return "low-centroid"

def compute_rhythm_metrics(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    if len(onsets) < 2:
        return 0, 0.0, 0.0
    
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    
    # Improved coherence: 1 - normalized std + entropy-based regularity
    cv = np.std(iois) / mean_ioi
    ioi_hist, _ = np.histogram(iois, bins=20)
    reg_entropy = 1 - (entropy(ioi_hist + 1e-10) / np.log(len(ioi_hist)))
    coherence = (1 - cv) * 0.5 + reg_entropy * 0.5
    coherence = np.clip(coherence, 0, 1)
    
    return len(onsets), mean_ioi, coherence

def compute_rhythm_lattice(iois, min_resolution=0.001):
    # Adaptive lattice base: approximate GCD of IOIs with clustering
    if len(iois) == 0:
        return 0.005, 1.0
    iois_sorted = np.sort(iois)
    diffs = np.diff(iois_sorted)
    base = np.gcd.reduce((iois * 1000).astype(int)) / 1000  # Millisecond precision
    base = max(base, min_resolution) if base > 0 else 0.005
    
    # Lattice coherence: fraction of IOIs that are multiples of base
    multiples = np.abs(iois / base - np.round(iois / base)) < 1e-3
    lattice_coherence = np.mean(multiples)
    
    return base, lattice_coherence

def compute_cqt_invariance(y, sr, n_bins=192):
    # CQT with tuned parameters for better invariance
    hop_length = 256 if sr == 22050 else int(sr / 100)  # Adaptive hop
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=24, filter_scale=1.5))
    
    # Improved shift invariance metric: average autocorrelation across bins
    invariance_scores = []
    for bin_idx in range(cqt.shape[0]):
        autocorr = correlate(cqt[bin_idx], cqt[bin_idx], mode='same')
        autocorr = autocorr / np.max(autocorr)
        # Measure peak width or decay; here, fraction above threshold
        invariance_scores.append(np.mean(autocorr > 0.5))
    
    metric = np.mean(invariance_scores)
    
    return cqt, metric

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    centroid_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_type} sound.")
    
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(y, sr)
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    iois = np.diff(librosa.onset.onset_detect(y=y, sr=sr, units='time')) if num_onsets >= 2 else np.array([])
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Adjust CQT params based on centroid for broad sound handling
    n_bins = 192 if centroid_type == "high-centroid" else 168
    cqt, invariance_metric = compute_cqt_invariance(y, sr, n_bins=n_bins)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    for file in wav_files:
        analyze_audio(file)