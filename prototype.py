# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, and CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to compute lattice coherence for a given base
def compute_lattice_coherence(onset_times, base):
    if len(onset_times) < 2:
        return 0.0
    start = onset_times[0]
    end = onset_times[-1]
    grid = np.arange(start, end + base, base)
    dists = [np.min(np.abs(t - grid)) for t in onset_times]
    tolerance = base * 0.1  # 10% tolerance
    coh = np.sum(np.array(dists) < tolerance) / len(onset_times)
    return coh

# Function to find optimal lattice base by maximizing coherence
def find_optimal_lattice_base(onset_times, iois):
    if len(onset_times) < 2:
        return np.mean(iois) if len(iois) > 0 else 0.0, 0.0
    # Possible bases: range from 0.05 to 1.0 s with finer steps for better precision
    possible_bases = np.arange(0.05, 1.01, 0.005)
    max_coh = 0.0
    best_base = np.median(iois)
    for b in possible_bases:
        coh = compute_lattice_coherence(onset_times, b)
        if coh > max_coh:
            max_coh = coh
            best_base = b
    return best_base, max_coh

# Function to analyze a single audio file
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Compute spectral centroid to classify sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 4000:
        print("  Detected high-centroid sound (e.g., birdsong).")
        print("  Using high-sensitivity onset params.")
        onset_params = {'backtrack': True, 'pre_max': 0.02, 'post_max': 0.02, 'wait': 0.01}
    elif mean_centroid < 1500:
        print("  Detected low-centroid sound (e.g., bass-heavy).")
        print("  Using low-sensitivity onset params.")
        onset_params = {'backtrack': False, 'pre_max': 0.05, 'post_max': 0.05, 'wait': 0.03}
    else:
        print("  Detected mid-centroid sound (e.g., orchestral or rock).")
        print("  Using standard-sensitivity onset params.")
        onset_params = {'backtrack': False, 'pre_max': 0.03, 'post_max': 0.03, 'wait': 0.02}
    
    # Onset detection with adaptive parameters
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    if len(onsets) < 2:
        print("  Insufficient onsets for rhythm analysis.")
        return
    
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    
    # Rhythm coherence: 1 - coefficient of variation (improved with clipping)
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = max(0, 1 - cv)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice: optimize base for max coherence
    lattice_base, lattice_coh = find_optimal_lattice_base(onsets, iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coh:.2f}")
    
    # Improved CQT: higher resolution for better invariance, adaptive fmin
    fmin = librosa.note_to_hz('C1') if mean_centroid < 1500 else librosa.note_to_hz('C2') if mean_centroid > 4000 else librosa.note_to_hz('A1')
    bins_per_octave = 36  # Increased for finer resolution and better shift invariance
    n_bins = 216  # 36 bpo * 6 octaves
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    # CQT shift invariance metric: normalized mean diff after 1-bin shift
    abs_cqt = np.abs(cqt)
    shifted = np.roll(abs_cqt, 1, axis=0)
    diff = np.mean(np.abs(abs_cqt - shifted))
    norm = np.mean(abs_cqt) + 1e-8  # Avoid division by zero
    metric = diff / norm
    print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")

# Main execution
print("Analyzing available WAV files.")
for file in available_files:
    if os.path.exists(file):
        print(f"Analysis for {file}:")
        analyze_audio(file)
    else:
        print(f"File {file} not found.")