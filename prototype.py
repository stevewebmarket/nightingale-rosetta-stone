# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import scipy.stats

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1500:
        return "low-centroid sound"
    elif mean_centroid < 4000:
        return "mid-centroid sound"
    else:
        return "high-centroid sound"

# Improved rhythm lattice calculation
def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.001, 0.0  # Default fallback
    iois = np.diff(onset_times)
    min_ioi = np.min(iois)
    base = min_ioi / 1000  # Finer granularity for lattice
    # Lattice coherence: improved as inverse of normalized std of IOIs projected to lattice
    lattice_points = np.round(iois / base)
    coherence = 1 - (np.std(lattice_points) / np.mean(lattice_points)) if np.mean(lattice_points) > 0 else 0
    return base, max(0, min(1, coherence))

# Improved rhythm coherence
def compute_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    # Use entropy as a measure of rhythm regularity (lower entropy = higher coherence)
    hist, _ = np.histogram(iois, bins=20)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = scipy.stats.entropy(prob)
    max_entropy = np.log(len(prob))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0

# Improved CQT with better shift invariance
def compute_cqt(y, sr):
    # Use smaller hop_length and more bins for better resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=256, n_bins=168, bins_per_octave=24)
    # Shift invariance metric: autocorrelation across time shifts
    cqt_mag = np.abs(cqt)
    autocorr = np.corrcoef(cqt_mag[:, :-1], cqt_mag[:, 1:])[0, 1]
    # Normalize and improve metric for broader sound handling
    invariance = (autocorr + 1) / 2  # Range 0 to 1, higher better
    return cqt, invariance

# Main analysis loop
for file in wav_files:
    y, sr = librosa.load(file, sr=22050)
    print(f"Analysis for {file}:")
    
    # Sound classification
    sound_type = classify_sound(y, sr)
    print(f"  Detected {sound_type}.")
    
    # Onset detection with backtracking for broader sound handling
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_times)}")
    
    # IOI and rhythm coherence
    if len(onset_times) >= 2:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        rhythm_coherence = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  mean IOI: N/A, rhythm coherence: N/A")
    
    # Rhythm lattice
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT analysis
    cqt, invariance = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")