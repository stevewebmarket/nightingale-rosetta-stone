# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Coherence + CQT Invariance + Broad Handling
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import correlate

# Define analysis function
def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        sound_type = "low-centroid sound"
    elif mean_centroid > 3000:
        sound_type = "high-centroid sound"
    else:
        sound_type = "mid-centroid sound"
    print(f"  Detected {sound_type}.")
    
    # Onset detection with improved sensitivity for broad sounds
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True, pre_max=0.05, post_max=0.05)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    print(f"  Detected onsets: {len(onsets)}")
    
    # Inter-onset intervals (IOIs) with improved rhythm lattice
    iois = np.diff(onset_times)
    if len(iois) == 0:
        iois = [0]
    mean_ioi = np.mean(iois)
    print(f"  mean IOI: {mean_ioi:.2f} s", end="")
    
    # Rhythm coherence: entropy-based for better coherence measure
    ioi_hist, _ = np.histogram(iois, bins=20)
    rhythm_coherence = 1 - entropy(ioi_hist + 1e-10) / np.log(len(ioi_hist))
    print(f", rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice base: autocorrelation for lattice detection
    if len(iois) > 1:
        autocorr = correlate(iois, iois, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks = np.where(autocorr > np.mean(autocorr) + np.std(autocorr))[0]
        if len(peaks) > 1:
            lattice_base = np.mean(np.diff(peaks)) * mean_ioi / len(peaks)
        else:
            lattice_base = mean_ioi / 2
    else:
        lattice_base = mean_ioi
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    
    # Lattice coherence: correlation strength
    lattice_coherence = np.max(autocorr) / np.sum(autocorr) if np.sum(autocorr) > 0 else 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with improved invariance: using HCQT-like approach for better shift invariance
    hop_length = 512
    n_bins = 384  # Increased for broader frequency coverage
    cqt = librosa.hybrid_cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, tuning=0.0, fmin=librosa.note_to_hz('C1'))
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    # Improved CQT shift invariance metric: average correlation across octave shifts
    invariance_scores = []
    for shift in range(1, 4):  # Check 1-3 octave shifts
        shifted_cqt = np.roll(cqt, shift * 12, axis=0)  # Shift by semitones in octave
        corr = np.corrcoef(cqt.flatten(), shifted_cqt.flatten())[0, 1]
        invariance_scores.append(corr)
    invariance_metric = np.mean(invariance_scores)
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

# List of available files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Analyze each file
for file in files:
    print(f"Analysis for {file}:")
    analyze_audio(file)
    print()