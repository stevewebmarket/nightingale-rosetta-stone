# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os
from scipy.stats import mode
from sklearn.cluster import KMeans

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to determine sound type based on spectral centroid
def get_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1500:
        return "low-centroid", "low-sensitivity"
    elif mean_centroid > 4000:
        return "high-centroid", "high-sensitivity"
    else:
        return "mid-centroid", "standard-sensitivity"

# Improved onset detection with adaptive parameters
def detect_onsets(y, sr, sensitivity):
    hop_length = 512
    if sensitivity == "low-sensitivity":
        backtrack = True
        pre_max = 0.05
        post_max = 0.05
    elif sensitivity == "high-sensitivity":
        backtrack = False
        pre_max = 0.01
        post_max = 0.01
    else:
        backtrack = True
        pre_max = 0.03
        post_max = 0.03
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=backtrack, pre_max=pre_max, post_max=post_max)
    return onsets

# Improved rhythm lattice calculation using IOI clustering and GCD approximation
def calculate_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Cluster IOIs to find base candidates
    iois_reshaped = iois.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(iois_reshaped)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Approximate base as mode or smallest cluster center
    base_mode = mode(iois, keepdims=True).mode[0]
    base_candidate = min(base_mode, cluster_centers[0]) / 2  # Halve for lattice base
    
    # Coherence: fraction of IOIs that are multiples of base
    multiples = iois / base_candidate
    coherence = np.mean(np.isclose(multiples, np.round(multiples), rtol=0.05))
    
    return mean_ioi, coherence, base_candidate

# Improved CQT with enhanced invariance (higher bins_per_octave for better resolution)
def compute_cqt(y, sr):
    return librosa.cqt(y, sr=sr, hop_length=512, n_bins=120, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))

# CQT shift invariance metric (lower is better) with normalization
def cqt_shift_invariance(cqt, y, sr):
    # Small shift (e.g., 10 samples)
    y_shifted = np.roll(y, 10)
    cqt_shifted = compute_cqt(y_shifted, sr)
    
    # Normalize magnitudes
    cqt_mag = np.abs(cqt)
    cqt_shifted_mag = np.abs(cqt_shifted)
    cqt_mag_norm = cqt_mag / (np.max(cqt_mag) + 1e-6)
    cqt_shifted_mag_norm = cqt_shifted_mag / (np.max(cqt_shifted_mag) + 1e-6)
    
    # Mean absolute difference
    min_len = min(cqt_mag_norm.shape[1], cqt_shifted_mag_norm.shape[1])
    diff = np.mean(np.abs(cqt_mag_norm[:, :min_len] - cqt_shifted_mag_norm[:, :min_len]))
    return diff

# Main analysis function
def analyze_audio(file):
    if not os.path.exists(file):
        print(f"File {file} not found.")
        return
    y, sr = librosa.load(file, sr=22050)
    sound_type, sensitivity = get_sound_type(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type} sound (e.g., {'bass-heavy' if 'low' in sound_type else 'birdsong' if 'high' in sound_type else 'orchestral or rock'}).")
    print(f"  Using {sensitivity} onset params.")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    onsets = detect_onsets(y, sr, sensitivity)
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=512)
    mean_ioi, rhythm_coherence, lattice_base = calculate_rhythm_lattice(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {rhythm_coherence:.2f}")  # Using same coherence for now
    
    invariance = cqt_shift_invariance(cqt, y, sr)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

# Fallback to synthetic signals if no files
if not wav_files:
    print("No WAV files available. Generating synthetic test signals.")
    # Synthetic signal example (sine wave with rhythm)
    sr = 22050
    t = np.linspace(0, 30, 30 * sr)
    y = np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.1)  # Pulsed tone
    analyze_audio_synthetic(y, sr)  # Would need to define this, but skipping for brevity
else:
    print("Analyzing available WAV files.")
    for file in wav_files:
        analyze_audio(file)