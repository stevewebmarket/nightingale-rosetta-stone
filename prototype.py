# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import os
import numpy as np
import librosa
import scipy.stats

# List of available WAV files in the current working directory
AVAILABLE_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to compute spectral centroid for sound type detection
def detect_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return 'high-centroid'  # e.g., birdsong, high-frequency dominant
    elif mean_centroid > 1500:
        return 'mid-centroid'  # e.g., orchestral, rock
    else:
        return 'low-centroid'  # e.g., bass-heavy or ambient

# Function to get onset detection parameters based on sound type
def get_onset_params(sound_type):
    if sound_type == 'high-centroid':
        return {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.05}  # High sensitivity
    elif sound_type == 'low-centroid':
        return {'backtrack': True, 'pre_max': 0.1, 'post_max': 0.1, 'delta': 0.2}  # Lower sensitivity for bass
    else:
        return {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.1}  # Standard

# Improved rhythm lattice: dynamic base period based on mean IOI
def compute_rhythm_lattice(onset_times, sr):
    if len(onset_times) < 2:
        return 0.005, 0.0  # Default fallback
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    base_period = mean_ioi / 16  # Subdivide into smaller units (improved granularity)
    # Lattice coherence: how well onsets align to the lattice (improved with entropy)
    lattice_times = np.arange(0, onset_times[-1], base_period)
    aligned = np.min(np.abs(onset_times[:, None] - lattice_times[None, :]), axis=1) / base_period
    coherence = 1 - scipy.stats.entropy(aligned + 1e-10) / np.log(len(lattice_times))  # Normalized entropy-based coherence
    return base_period, max(0, min(1, coherence))  # Clamp between 0 and 1

# Improved rhythm coherence: using autocorrelation consistency
def compute_rhythm_coherence(onset_times):
    if len(onset_times) < 2:
        return 0.0
    iois = np.diff(onset_times)
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    coherence = np.mean(autocorr / np.max(autocorr))  # Normalized autocorrelation mean (improved)
    return max(0, min(1, coherence))

# Improved CQT with better shift invariance: use smaller hop_length and compute invariance metric
def compute_cqt_and_invariance(y, sr):
    hop_length = 256  # Smaller hop for better time resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12, filter_scale=1.0)
    # Shift invariance metric: average correlation with small time-shifted versions (improved)
    shifts = [1, 2, 3]  # Pixel shifts
    invariance = 0
    for shift in shifts:
        shifted = np.roll(cqt, shift, axis=1)
        corr = np.mean(np.abs(np.corrcoef(cqt.flatten(), shifted.flatten())))
        invariance += (1 - corr)  # Difference from perfect correlation (lower is more invariant to shifts)
    invariance /= len(shifts)
    return cqt, invariance

# Main analysis function
def analyze_audio(file_path):
    print(f"Analysis for {file_path}:")
    y, sr = librosa.load(file_path, sr=22050)
    
    sound_type = detect_sound_type(y, sr)
    if sound_type == 'high-centroid':
        print("  Detected high-centroid sound (e.g., birdsong).")
        print("  Using high-sensitivity onset params.")
    elif sound_type == 'low-centroid':
        print("  Detected low-centroid sound (e.g., bass-heavy).")
        print("  Using low-sensitivity onset params.")
    else:
        print("  Detected mid-centroid sound (e.g., orchestral or rock).")
        print("  Using standard onset params.")
    
    onset_params = get_onset_params(sound_type)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    cqt, invariance_metric = compute_cqt_and_invariance(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  Detected onsets: {len(onset_times)}")
    
    if len(onset_times) > 1:
        mean_ioi = np.mean(np.diff(onset_times))
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(onset_times):.2f}")
    else:
        print("  mean IOI: N/A, rhythm coherence: 0.00")
    
    base_period, lattice_coherence = compute_rhythm_lattice(onset_times, sr)
    print(f"  Rhythm lattice base period u1e6: Rhythm lattice base: {base_period:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

# Synthetic test signal fallback
def generate_synthetic_signal():
    sr = 22050
    t = np.linspace(0, 5, int(5 * sr), endpoint=False)
    y = np.sin(440 * 2 * np.pi * t) + 0.5 * np.sin(880 * 2 * np.pi * t)
    return y, sr

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    if AVAILABLE_FILES:
        for file in AVAILABLE_FILES:
            if os.path.exists(file):
                analyze_audio(file)
            else:
                print(f"File {file} not found, skipping.")
    else:
        print("No WAV files available, using synthetic test signal.")
        y, sr = generate_synthetic_signal()
        # Save to temp file or analyze directly
        analyze_audio('synthetic')  # Placeholder, adapt as needed