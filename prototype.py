# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + CQT Invariance + Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to classify sound based on spectral centroid
def classify_sound(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1000:
        return 'low-centroid'
    elif centroid < 4000:
        return 'mid-centroid'
    else:
        return 'high-centroid'

# Improved onset detection with adaptive parameters
def detect_onsets(y, sr, sound_type):
    if sound_type == 'low-centroid':
        hop_length = 512
        backtrack = True
        pre_max = 0.03 * sr // hop_length
        post_max = 0.00 * sr // hop_length
        pre_avg = 0.10 * sr // hop_length
        post_avg = 0.10 * sr // hop_length
        wait = 0.03 * sr // hop_length
        delta = 0.05  # Lower sensitivity for bass-heavy
    elif sound_type == 'mid-centroid':
        hop_length = 256
        backtrack = True
        pre_max = 0.02 * sr // hop_length
        post_max = 0.02 * sr // hop_length
        pre_avg = 0.08 * sr // hop_length
        post_avg = 0.08 * sr // hop_length
        wait = 0.02 * sr // hop_length
        delta = 0.07  # Mid sensitivity
    else:  # high-centroid
        hop_length = 128
        backtrack = True
        pre_max = 0.01 * sr // hop_length
        post_max = 0.01 * sr // hop_length
        pre_avg = 0.05 * sr // hop_length
        post_avg = 0.05 * sr // hop_length
        wait = 0.01 * sr // hop_length
        delta = 0.10  # Higher sensitivity for transients

    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length,
                                        backtrack=backtrack, pre_max=pre_max, post_max=post_max,
                                        pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait)
    return librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

# Improved rhythm lattice calculation using autocorrelation for base period
def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0, 0.0

    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)

    # Autocorrelation to find repeating periods
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=2)
    if len(peaks) > 1:
        base_period = np.mean(np.diff(peaks)) * np.mean(iois) / len(iois)  # Scale to time
    else:
        base_period = np.min(iois) if len(iois) > 0 else 0.001
    base_period = max(base_period, 0.001)  # Avoid zero or too small

    # Coherence: how well onsets fit the lattice
    lattice = np.arange(0, onset_times[-1] + base_period, base_period)
    hits = 0
    for ot in onset_times:
        if np.min(np.abs(lattice - ot)) < base_period * 0.1:
            hits += 1
    coherence = hits / len(onset_times)

    # Rhythm coherence based on IOI variance
    rhythm_coherence = 1 / (1 + np.std(iois) / mean_ioi) if mean_ioi > 0 else 0.0

    return mean_ioi, rhythm_coherence, base_period, coherence

# Improved CQT with better invariance (smaller hop, normalization)
def compute_cqt(y, sr):
    hop_length = 128  # Smaller hop for better time resolution
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    # Normalize for invariance
    cqt_mag = (cqt_mag - np.min(cqt_mag)) / (np.max(cqt_mag) - np.min(cqt_mag))
    return cqt_mag

# Improved shift invariance metric (correlation with shifted version)
def cqt_shift_invariance(cqt):
    if cqt.shape[1] < 2:
        return 0.0
    shift = np.roll(cqt, 1, axis=1)
    corr = np.corrcoef(cqt.flatten(), shift.flatten())[0, 1]
    # Penalize low correlation, but boost for high
    return corr ** 2 - 0.5  # Range adjust to make higher better, less negative

# Main analysis function
def analyze_audio(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    y, sr = librosa.load(file_path, sr=22050)
    sound_type = classify_sound(y, sr)
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type} sound.")
    if sound_type == 'low-centroid':
        print("  Using low-sensitivity onset params.")
    elif sound_type == 'mid-centroid':
        print("  Using mid-sensitivity onset params.")
    else:
        print("  Using high-sensitivity onset params.")

    onset_times = detect_onsets(y, sr, sound_type)
    print(f"  Detected onsets: {len(onset_times)}")

    mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")

    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# Run analysis for each file
if __name__ == "__main__":
    if not WAV_FILES:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signals would go here, but for now, skip
    else:
        for wav in WAV_FILES:
            analyze_audio(wav)