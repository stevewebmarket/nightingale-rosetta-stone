# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import os
import numpy as np
import librosa
import librosa.feature

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        return 'high-centroid'
    elif centroid > 2000:
        return 'mid-centroid'
    else:
        return 'low-centroid'

# Function to adjust onset detection parameters based on classification
def get_onset_params(sound_type):
    if sound_type == 'high-centroid':
        return {'delta': 0.02, 'wait': 1, 'pre_max': 0.01, 'post_max': 0.01}  # High sensitivity
    elif sound_type == 'mid-centroid':
        return {'delta': 0.05, 'wait': 2, 'pre_max': 0.03, 'post_max': 0.03}  # Mid sensitivity
    else:
        return {'delta': 0.1, 'wait': 4, 'pre_max': 0.06, 'post_max': 0.06}   # Low sensitivity

# Function to compute rhythm coherence (improved: using autocorrelation consistency)
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    # Normalized autocorrelation for periodicity measure
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr /= autocorr[0]  # Normalize
    coherence = np.mean(autocorr[1:]) if len(autocorr) > 1 else 0.0
    return max(0.0, min(1.0, coherence))

# Function to compute adaptive rhythm lattice base
def compute_lattice_base(mean_ioi):
    return max(0.001, mean_ioi / 20.0)  # Adaptive, finer for faster rhythms

# Function to compute lattice coherence (simplified entropy-based)
def compute_lattice_coherence(onset_times, lattice_base):
    if len(onset_times) == 0:
        return 0.0
    lattice_points = np.round(onset_times / lattice_base)
    unique_points = np.unique(lattice_points)
    probs = np.array([np.sum(lattice_points == p) for p in unique_points]) / len(lattice_points)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(unique_points))
    coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    return coherence

# Function to compute CQT with improved parameters for invariance
def compute_cqt(y, sr):
    # Increased bins per octave for better frequency resolution and invariance
    return librosa.cqt(y, sr=sr, hop_length=512, n_bins=84*6, bins_per_octave=84, fmin=librosa.note_to_hz('C1'))

# Function to compute CQT shift invariance metric (lower is better)
def cqt_shift_invariance(cqt):
    # Simulate small time shift and compute normalized difference
    shifted = np.roll(cqt, 1, axis=1)
    diff = np.mean(np.abs(cqt[:, :shifted.shape[1]] - shifted[:, :shifted.shape[1]]))
    norm = np.mean(np.abs(cqt)) + 1e-10
    return diff / norm

# Main analysis loop
for wav in wav_files:
    if not os.path.exists(wav):
        print(f"File {wav} not found, skipping.")
        continue
    
    print(f"Analysis for {wav}:")
    y, sr = librosa.load(wav, sr=22050)
    
    # Classify sound
    sound_type = classify_sound(y, sr)
    print(f"  Detected {sound_type} sound.")
    
    # Get onset params
    params = get_onset_params(sound_type)
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    # Onset detection with adaptive params
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, delta=params['delta'], wait=params['wait'], pre_max=params['pre_max'], post_max=params['post_max'])
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_times)}")
    
    # Compute IOIs and mean
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
    
    # Adaptive rhythm lattice
    lattice_base = compute_lattice_base(mean_ioi)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {compute_lattice_coherence(onset_times, lattice_base):.2f}")
    
    # Compute CQT
    cqt = np.abs(compute_cqt(y, sr))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Invariance metric
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")