# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np
import scipy.stats

# List of available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def detect_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return "high-centroid sound"
    elif mean_centroid > 2000:
        return "mid-centroid sound"
    else:
        return "low-centroid sound"

def compute_rhythm_lattice(iois):
    if len(iois) < 2:
        return 0.2  # Default fallback
    # Improved: Use mode of IOIs for base lattice period, with histogram binning
    hist, bin_edges = np.histogram(iois, bins=50)
    mode_bin = bin_edges[np.argmax(hist)]
    return mode_bin

def compute_coherence(iois, lattice_base):
    if len(iois) < 2 or lattice_base == 0:
        return 0.0
    # Improved: Fraction of IOIs that are multiples of lattice_base within tolerance
    tolerance = 0.05 * lattice_base
    aligned = np.abs(iois % lattice_base) < tolerance
    return np.mean(aligned)

def compute_lattice_coherence(onset_times, lattice_base):
    if len(onset_times) < 2 or lattice_base == 0:
        return 0.0
    # Improved: Alignment of all onsets to lattice grid
    grid = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
    aligned_counts = 0
    for ot in onset_times:
        if np.min(np.abs(ot - grid)) < 0.05 * lattice_base:
            aligned_counts += 1
    return aligned_counts / len(onset_times)

def compute_cqt_invariance(cqt):
    # Improved: Measure shift invariance by average correlation between frames and shifted versions
    # Also added normalization for better invariance across sound types
    cqt_norm = librosa.util.normalize(np.abs(cqt), axis=0)
    correlations = []
    for shift in [1, 2, 3]:  # Check small shifts
        shifted = np.roll(cqt_norm, shift, axis=1)
        corr = np.mean([scipy.stats.pearsonr(cqt_norm[:, i], shifted[:, i])[0] for i in range(cqt_norm.shape[1])])
        correlations.append(corr)
    return np.mean(correlations)

for file in files:
    print(f"Analysis for {file}:")
    
    # Load audio with broader handling: mono=True for consistency, duration=None to load full
    y, sr = librosa.load(file, sr=22050, mono=True)
    
    # Detect sound type based on spectral centroid
    sound_type = detect_sound_type(y, sr)
    print(f"  Detected {sound_type}.")
    
    # Onset detection with backtrack for better accuracy on broad sounds
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_times)}")
    
    # Inter-onset intervals (IOIs)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_coherence(iois, mean_ioi):.2f}")
    
    # Improved adaptive rhythm lattice base
    lattice_base = compute_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {compute_lattice_coherence(onset_times, lattice_base):.2f}")
    
    # CQT with improved parameters for invariance: more bins per octave, adjusted hop_length based on sound type
    hop_length = 256 if "low" in sound_type else 512  # Adjust for broader handling
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)  # Increased bins/octave for better invariance
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Compute improved invariance metric
    invariance_metric = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")
    print()