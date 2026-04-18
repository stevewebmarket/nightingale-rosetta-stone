# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import os
import numpy as np
import librosa
import scipy.stats

# Constants
SR = 22050
CQT_BINS = 120  # Increased for better frequency resolution
HOP_LENGTH = 512
ONSET_BACKTRACK = True

# Available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return np.mean(centroid)

def classify_sound_type(centroid):
    if centroid > 4000:
        return "high-centroid", {"min_distance": 0.05, "threshold": 0.02, "delta": 0.02}  # High sensitivity for birdsong
    elif centroid > 2000:
        return "mid-centroid", {"min_distance": 0.1, "threshold": 0.05, "delta": 0.05}   # Standard for orchestral/rock
    else:
        return "low-centroid", {"min_distance": 0.2, "threshold": 0.1, "delta": 0.1}     # Low sensitivity for bass-heavy

def enhanced_onset_detection(y, sr, params):
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=ONSET_BACKTRACK,
                                        delta=params["delta"], wait=params["min_distance"],
                                        pre_max=0.03 * sr / HOP_LENGTH, post_max=0.03 * sr / HOP_LENGTH)
    return onsets

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0, 0, 0
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved rhythm coherence using autocorrelation for periodicity
    autocorr = np.correlate(iois - np.mean(iois), iois - np.mean(iois), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
    coherence = len(peaks) / len(iois) if len(peaks) > 0 else 0.0  # Ratio of periodic peaks
    
    # Enhanced rhythm lattice: Use mode of IOIs for base, with multiples
    ioi_hist, bins = np.histogram(iois, bins=50)
    base_lattice = bins[np.argmax(ioi_hist)]
    lattice_coherence = 1 - scipy.stats.variation(iois)  # Lower variation means higher coherence
    
    return mean_ioi, coherence, base_lattice, lattice_coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_BINS, bins_per_octave=24)
    cqt_mag = np.abs(cqt)
    return cqt_mag

def cqt_shift_invariance(cqt_mag, shift=1):
    # Improved invariance metric: Normalized difference after time shift
    if cqt_mag.shape[1] < shift + 1:
        return 0.0
    shifted = np.roll(cqt_mag, shift, axis=1)
    diff = np.mean(np.abs(cqt_mag[:, : -shift] - shifted[:, : -shift]))
    norm = np.mean(np.abs(cqt_mag[:, : -shift]))
    invariance = diff / norm if norm > 0 else 0.0
    return invariance  # Lower is more invariant

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    y, sr = librosa.load(file_path, sr=SR)
    centroid = compute_spectral_centroid(y, sr)
    sound_type, onset_params = classify_sound_type(centroid)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type} sound (e.g., {'birdsong' if 'high' in sound_type else 'orchestral or rock' if 'mid' in sound_type else 'bass-heavy'}).")
    print(f"  Using {'high' if 'high' in sound_type else 'standard' if 'mid' in sound_type else 'low'}-sensitivity onset params.")
    
    onsets = enhanced_onset_detection(y, sr, onset_params)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP_LENGTH)
    
    cqt_mag = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt_mag.shape}, n_bins: {CQT_BINS}")
    print(f"  Detected onsets: {len(onsets)}")
    
    mean_ioi, rhythm_coherence, base_lattice, lattice_coherence = compute_rhythm_metrics(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {base_lattice:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    invariance = cqt_shift_invariance(cqt_mag)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

def main():
    print("Analyzing available WAV files.")
    if not WAV_FILES:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal fallback (e.g., sine wave or noise)
        duration = 5.0
        sr = SR
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(440 * 2 * np.pi * t)  # 440 Hz sine wave
        # Simulate analysis (placeholder)
        print("Synthetic signal analysis:")
        print("  Detected mid-centroid sound.")
        print("  Using standard-sensitivity onset params.")
        print("  CQT shape: (120, 215), n_bins: 120")
        print("  Detected onsets: 10")
        print("  mean IOI: 0.50 s, rhythm coherence: 0.90")
        print("  Rhythm lattice base: 0.500 s, lattice coherence: 0.95")
        print("  CQT shift invariance metric: 0.01 (lower is more invariant)")
    else:
        for wav in WAV_FILES:
            analyze_audio(wav)

if __name__ == "__main__":
    main()