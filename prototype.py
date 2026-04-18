# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Improved CQT Invariance
# =============================================================================

import os
import numpy as np
import librosa

# Constants
SAMPLE_RATE = 22050
CQT_HOP_LENGTH = 512
MIN_FREQ = 27.5  # A0
BINS_PER_OCTAVE = 24

# Available WAV files
AVAILABLE_WAVS = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def detect_sound_type(y, sr):
    """Detect sound type based on spectral centroid."""
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 3000:
        return 'high-centroid'  # e.g., birdsong
    elif centroid > 1000:
        return 'mid-centroid'  # e.g., orchestral or rock
    else:
        return 'low-centroid'  # e.g., bass-heavy

def adjust_cqt_params(sound_type):
    """Adjust CQT parameters based on sound type."""
    if sound_type == 'high-centroid':
        return 120, MIN_FREQ / 2  # More bins, lower min freq for high freq content
    elif sound_type == 'mid-centroid':
        return 96, MIN_FREQ
    else:
        return 72, MIN_FREQ * 2  # Fewer bins, higher min freq for low freq

def compute_cqt(y, sr, n_bins, fmin):
    """Compute CQT with invariance enhancements."""
    cqt = librosa.cqt(y, sr=sr, hop_length=CQT_HOP_LENGTH, fmin=fmin, n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))  # Log amplitude for better invariance
    return log_cqt

def enhanced_onset_detection(y, sr, sound_type):
    """Enhanced onset detection with sound-type adjustments."""
    if sound_type == 'high-centroid':
        oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
    elif sound_type == 'mid-centroid':
        oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, fmax=4000, n_mels=128)
    else:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, fmax=2000, n_mels=64)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=True, pre_max=0.02, post_max=0.02)
    return onsets

def calculate_rhythm_metrics(onsets, sr):
    """Calculate rhythm metrics with improved coherence."""
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    # Rhythm coherence: lower std dev of IOIs indicates higher coherence
    rhythm_coherence = 1 / (1 + np.std(iois) / (mean_ioi + 1e-5)) if len(iois) > 1 else 0
    return mean_ioi, rhythm_coherence

def improved_rhythm_lattice(onsets, sr):
    """Improved rhythm lattice using autocorrelation for better base period."""
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    if len(iois) < 2:
        return 0, 0
    # Autocorrelation to find fundamental period
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
    base = np.mean(np.diff(peaks)) * np.mean(iois) / len(iois) if len(peaks) > 1 else np.min(iois)
    # Lattice coherence: fit to grid
    grid = np.arange(0, times[-1], base)
    fits = [np.min(np.abs(t - grid)) for t in times]
    lattice_coherence = 1 / (1 + np.mean(fits) / base) if base > 0 else 0
    return base, lattice_coherence

def cqt_shift_invariance(cqt):
    """Compute CQT shift invariance metric (lower is better)."""
    # Simulate small shifts and compute difference
    shifted = np.roll(cqt, 1, axis=1)
    diff = np.mean(np.abs(cqt[:, 1:] - shifted[:, 1:])) / np.mean(np.abs(cqt)) + 1e-5
    return diff

def analyze_audio(file_path):
    """Analyze a single audio file."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    sound_type = detect_sound_type(y, sr)
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type} sound (e.g., {'birdsong' if sound_type == 'high-centroid' else 'orchestral or rock'}), using adjusted onset detection.")
    
    n_bins, fmin = adjust_cqt_params(sound_type)
    cqt = compute_cqt(y, sr, n_bins, fmin)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    onsets = enhanced_onset_detection(y, sr, sound_type)
    print(f"  Detected onsets: {len(onsets)}, ", end='')
    
    mean_ioi, rhythm_coherence = calculate_rhythm_metrics(onsets, sr)
    print(f"mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    base, lattice_coherence = improved_rhythm_lattice(onsets, sr)
    print(f"  Rhythm lattice base: {base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

def main():
    print("Analyzing available WAV files.")
    if not AVAILABLE_WAVS:
        print("No WAV files available, falling back to synthetic test signals.")
        # Synthetic signal fallback (simple tone)
        duration = 5.0
        sr = SAMPLE_RATE
        y = librosa.tone(440, sr=sr, duration=duration)
        np.save('synthetic.npy', y)  # Dummy save
        analyze_audio('synthetic.npy')  # But treat as if WAV
    else:
        for wav in AVAILABLE_WAVS:
            analyze_audio(wav)

if __name__ == "__main__":
    main()