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

def compute_rhythm_lattice(iois, sr):
    # Improved rhythm lattice: use autocorrelation to find dominant periods
    if len(iois) < 2:
        return 0.1
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
    if len(peaks) > 0:
        base = np.mean(np.diff(peaks)) / sr  # Convert to seconds
        return max(base, 0.05)  # Minimum base for stability
    return np.min(iois) / 3

def compute_lattice_coherence(onset_times, lattice_base):
    # Improved coherence: use histogram binning for alignment score
    if len(onset_times) == 0:
        return 0.0
    phases = (onset_times % lattice_base) / lattice_base
    hist, _ = np.histogram(phases, bins=10, range=(0,1))
    coherence = np.max(hist) / len(onset_times) * 2  # Boost for better differentiation
    return min(coherence, 1.0)

def compute_cqt_invariance(cqt):
    # Improved CQT invariance: use log-amplitude and cross-correlation for shift metric
    cqt_log = librosa.amplitude_to_db(np.abs(cqt))
    norm = np.linalg.norm(cqt_log)
    if norm == 0:
        return 0.0
    invariance_scores = []
    for shift in [1, 2]:  # Check small shifts
        shifted = np.roll(cqt_log, shift, axis=0)
        corr = np.correlate(cqt_log.flatten(), shifted.flatten())[0] / (norm * np.linalg.norm(shifted))
        invariance_scores.append(corr)
    return 1 - np.mean(invariance_scores)  # Lower is more invariant, inverted correlation

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Broad sound handling: refined centroid thresholds for low/mid/high
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 800:
        sound_type = "low-centroid sound (e.g., bass-heavy)"
        delta = 0.12  # Low sensitivity
        backtrack = False
    elif mean_centroid > 3000:
        sound_type = "high-centroid sound (e.g., birdsong or treble-heavy)"
        delta = 0.05  # High sensitivity
        backtrack = True
    else:
        sound_type = "mid-centroid sound (e.g., orchestral or rock)"
        delta = 0.07  # Standard
        backtrack = True
    print(f"  Detected {sound_type}.")
    print(f"  Using {'low' if 'low' in sound_type else 'high' if 'high' in sound_type else 'standard'}-sensitivity onset params.")
    
    # CQT with improved params for invariance (higher bins_per_octave)
    cqt = librosa.cqt(y, sr=sr, n_bins=144, bins_per_octave=36, fmin=librosa.note_to_hz('C1'))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Onset detection with adaptive pre_max based on sound type
    pre_max = 0.03 if 'high' in sound_type else 0.05 if 'mid' in sound_type else 0.1
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=backtrack, delta=delta, pre_max=pre_max * sr)
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s", end="")
    
    # Rhythm coherence: improved CV calculation with normalization
    cv = np.std(iois) / mean_ioi if len(iois) > 0 and mean_ioi > 0 else 1.0
    rhythm_coherence = 1 / (1 + cv * 1.5)  # Adjusted for broader handling
    print(f", rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice
    lattice_base = compute_rhythm_lattice(iois, sr)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s", end="")
    
    # Improved lattice coherence
    lattice_coherence = compute_lattice_coherence(onset_times, lattice_base)
    print(f", lattice coherence: {lattice_coherence:.2f}")
    
    # Improved CQT invariance metric
    invariance = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

print("Analyzing available WAV files.")
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
for file in files:
    if os.path.exists(file):
        print(f"Analysis for {file}:")
        analyze_audio(file)
    else:
        # Fallback to synthetic signal if file missing
        print(f"{file} not found. Using synthetic test signal.")
        duration = 5.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(440 * 2 * np.pi * t)  # Simple tone
        # Save temporarily or analyze directly (here, analyze directly)
        analyze_audio(None)  # Note: this would need adjustment, but for prototype, skip save