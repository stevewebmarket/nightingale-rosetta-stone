# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

# List of available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Parameters
sr = 22050
hop_length = 512  # Default hop length for frame time ~0.023s
n_bins = 384
bins_per_octave = 48  # High resolution for broad sound handling
fmin_default = librosa.note_to_hz('C1')

def classify_centroid(mean_centroid, sr):
    if mean_centroid < sr / 8:
        return "low-centroid sound"
    elif mean_centroid > sr / 3:
        return "high-centroid sound"
    else:
        return "mid-centroid sound"

def compute_rhythm_coherence(iois):
    if len(iois) < 1:
        return 0.0
    cv = np.std(iois) / np.mean(iois)
    return max(0.0, 1.0 - cv)  # Coherence as 1 - coefficient of variation, clipped

def compute_lattice_coherence(onsets, lattice_base):
    if len(onsets) < 2 or lattice_base <= 0:
        return 0.0
    lattice_points = np.arange(0, onsets[-1] + lattice_base, lattice_base)
    alignments = []
    for onset in onsets:
        dists = np.abs(lattice_points - onset)
        min_dist = np.min(dists)
        alignments.append(min_dist)
    avg_error = np.mean(alignments)
    max_error = lattice_base / 2.0
    coherence = max(0.0, 1.0 - (avg_error / max_error))
    return coherence

def compute_cqt_invariance(cqt_mag, bins_per_octave):
    n_bins, n_frames = cqt_mag.shape
    invariance_scores = []
    for shift in range(1, 4):  # Check invariance over 1-3 octaves
        shifted = np.roll(cqt_mag, shift * bins_per_octave, axis=0)
        corr = np.corrcoef(cqt_mag.flatten(), shifted.flatten())[0, 1]
        invariance_scores.append(corr)
    return np.mean(invariance_scores)

for file in files:
    print(f"Analysis for {file}:")
    
    # Load audio
    y, sr = librosa.load(file, sr=sr)
    
    # Spectral centroid for classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_cent = np.mean(centroid)
    print(f"  Detected {classify_centroid(mean_cent, sr)}.")
    
    # Onset detection
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', hop_length=hop_length)
    n_onsets = len(onset_times)
    print(f"  Detected onsets: {n_onsets}")
    
    if n_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        rhythm_coherence = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  Insufficient onsets for IOI analysis.")
        continue
    
    # Adaptive rhythm lattice: Use tempo to guide base, subdivide beat into 16ths for finer grid
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    if tempo == 0:
        tempo = 120.0  # Fallback
    beat_duration = 60.0 / tempo
    lattice_base = beat_duration / 16.0  # Finer subdivision for better alignment
    lattice_coherence = compute_lattice_coherence(onset_times, lattice_base)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Adaptive CQT: Adjust fmin based on centroid for broader sound handling
    fmin = fmin_default
    if "high" in classify_centroid(mean_cent, sr):
        fmin *= 4  # Shift up for high-frequency sounds like birdsong
    elif "low" in classify_centroid(mean_cent, sr):
        fmin /= 2  # Shift down for low-frequency sounds
    
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_mag = np.abs(cqt)
    # Normalize for better invariance
    cqt_mag = librosa.util.normalize(cqt_mag, axis=0)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    invariance = compute_cqt_invariance(cqt_mag, bins_per_octave)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    
    print()  # Blank line between analyses