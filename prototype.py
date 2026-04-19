# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Dynamic Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import correlate

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 3000:
        return "high-centroid sound"
    elif centroid > 1000:
        return "mid-centroid sound"
    else:
        return "low-centroid sound"

# Improved rhythm lattice calculation with dynamic base
def compute_rhythm_lattice(onsets, sr):
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    if len(iois) == 0:
        return 0.0, 0.0, 0.0
    
    mean_ioi = np.mean(iois)
    # Dynamic base: use gcd approximation for lattice base
    iois_sorted = np.sort(iois)
    gcd_approx = np.gcd.reduce((iois_sorted * sr).astype(int)) / sr if len(iois_sorted) > 1 else mean_ioi
    lattice_base = max(0.05, min(gcd_approx, mean_ioi / 2))  # Constrain for stability
    
    # Improved coherence: fraction of onsets fitting lattice within tolerance
    tolerance = lattice_base * 0.1
    fitted = np.sum(np.abs(onset_times % lattice_base) < tolerance) / len(onset_times)
    rhythm_coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0.0
    lattice_coherence = fitted * rhythm_coherence  # Combined metric
    
    return mean_ioi, rhythm_coherence, lattice_base, lattice_coherence

# Enhanced CQT with improved shift invariance
def compute_cqt_metrics(y, sr):
    # Use HCQT-like approach for better invariance: multiple octaves
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=168, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))
    
    # Normalize for invariance
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    cqt_norm = (cqt_mag - np.min(cqt_mag)) / (np.max(cqt_mag) - np.min(cqt_mag) + 1e-6)
    
    # Improved shift invariance metric: auto-correlation across bins
    autocorr = np.mean([np.max(correlate(cqt_norm[i], cqt_norm[i], mode='full')) for i in range(cqt_norm.shape[0])])
    invariance = autocorr / (cqt_norm.shape[1] * np.mean(np.abs(cqt_norm)))  # Normalized, higher is better
    invariance = np.clip(invariance, 0, 1)  # Clip to [0,1]
    
    return cqt.shape, cqt.shape[0], invariance

# Main analysis function
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    sound_type = classify_sound(y, sr)
    
    # Onset detection with backtracking for broader sound handling
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, units='frames')
    num_onsets = len(onsets)
    
    mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_lattice(onsets, sr)
    
    cqt_shape, n_bins, invariance = compute_cqt_metrics(y, sr)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type}.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Run analysis for each file
for file in wav_files:
    analyze_audio(file)