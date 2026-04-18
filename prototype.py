# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Refinement + CQT Invariance Boost
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans

# Constants
SR = 22050
HOP_LENGTH = 512
N_BINS = 384  # Increased for finer frequency resolution
MIN_FREQ = 20.0
N_OCTAVES = 8

# Available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return "high-centroid"
    elif mean_centroid > 2000:
        return "mid-centroid"
    else:
        return "low-centroid"

def detect_onsets(y, sr, hop_length=512):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    return onset_times

def compute_iois(onset_times):
    iois = np.diff(onset_times)
    iois = iois[iois > 0]
    return iois

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, bin_edges = np.histogram(iois, bins=50)
    hist = hist / hist.sum()
    return 1 - entropy(hist) / np.log(len(hist))

def refined_rhythm_lattice(iois, n_clusters=3):
    if len(iois) < 2:
        return 0.0, 0.0
    iois = iois.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(iois)
    base = np.mean(kmeans.cluster_centers_)
    labels = kmeans.labels_
    coherence = max([np.sum(labels == i) / len(labels) for i in range(n_clusters)])
    return base, coherence

def compute_cqt(y, sr, hop_length=512, n_bins=N_BINS, min_freq=MIN_FREQ):
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, fmin=min_freq, bins_per_octave=n_bins//N_OCTAVES)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    return cqt_mag

def cqt_shift_invariance(cqt, shifts=[1, 2, 4, 8]):
    invariance_scores = []
    for shift in shifts:
        rolled = np.roll(cqt, shift, axis=1)
        diff = np.mean(np.abs(cqt - rolled))
        invariance_scores.append(1 / (1 + diff))  # Normalized invariance
    return np.mean(invariance_scores)

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    centroid_type = compute_spectral_centroid(y, sr)
    onset_times = detect_onsets(y, sr, HOP_LENGTH)
    iois = compute_iois(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    coh = rhythm_coherence(iois)
    lattice_base, lattice_coh = refined_rhythm_lattice(iois)
    cqt = compute_cqt(y, sr, HOP_LENGTH)
    invariance = cqt_shift_invariance(cqt)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_type} sound.")
    print(f"  Detected onsets: {len(onset_times)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coh:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coh:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    for wav in WAV_FILES:
        analyze_audio(wav)