# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Detection, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Fallback synthetic signals if no files
if not wav_files:
    # Synthetic test signal: simple sine wave with rhythm
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.1)  # Pulsed tone
    filenames = ['synthetic.wav']
    # Note: In real scenario, save to file, but here simulate
else:
    filenames = wav_files

def compute_rhythm_lattice(onset_times, sr):
    if len(onset_times) < 2:
        return 0.0, 0.0
    # Improved: Use autocorrelation to find dominant period
    iois = np.diff(onset_times)
    lags = np.arange(1, min(20, len(iois)))
    autocorr = np.array([np.corrcoef(iois[:-lag], iois[lag:])[0,1] for lag in lags])
    dominant_lag = lags[np.argmax(autocorr)] if np.max(autocorr) > 0.1 else 1
    base = np.mean(iois[::dominant_lag])
    # Generate lattice grid
    grid = np.arange(0, onset_times[-1] + base, base)
    # Coherence: average min distance to grid, normalized
    distances = np.min([np.abs(ot - grid) for ot in onset_times], axis=1)
    coherence = 1 - np.mean(distances) / (base / 2)
    coherence = max(0, coherence)
    return base, coherence

def compute_cqt_invariance(cqt, hop_length):
    # Improved: Use magnitude CQT, compute average shift correlation over small shifts
    mag_cqt = np.abs(cqt)
    invariance = 0
    num_shifts = 5
    for shift in range(1, num_shifts + 1):
        shifted = np.roll(mag_cqt, shift, axis=1)
        corr = np.corrcoef(mag_cqt.flatten(), shifted.flatten())[0,1]
        invariance += corr
    invariance /= num_shifts
    # Adjust for broad sounds: normalize by temporal autocorrelation
    autocorr = np.corrcoef(mag_cqt[:, :-1].flatten(), mag_cqt[:, 1:].flatten())[0,1]
    invariance = (invariance + autocorr) / 2
    return invariance

def analyze_audio(filename, sr=22050):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return
    y, sr = librosa.load(filename, sr=sr)
    
    # Spectral centroid for sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    sound_type = "high-centroid" if mean_centroid > 3000 else "mid-centroid"  # Threshold for broad handling
    
    # Onsets detection with backtracking for better accuracy on broad sounds
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    num_onsets = len(onset_times)
    
    # Mean IOI
    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        # Rhythm coherence: 1 - coefficient of variation, clipped
        cv = np.std(iois) / mean_ioi
        rhythm_coherence = max(0, 1 - cv)
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    
    # Improved rhythm lattice
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times, sr)
    
    # CQT with adjusted parameters for broad sounds
    hop_length = 512 if sound_type == "mid-centroid" else 256  # Smaller hop for high-centroid
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]
    
    # Improved CQT shift invariance
    invariance_metric = compute_cqt_invariance(cqt, hop_length)
    
    # Output
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    for fn in filenames:
        if 'synthetic' in fn:
            # Simulate synthetic analysis
            print("Using synthetic signal.")
            # Define y, sr here if needed, but for demo, skip to analyze
            # In real, save to file, but here call directly if needed
            pass  # Placeholder; in code, would generate y and analyze without file
        else:
            analyze_audio(fn)