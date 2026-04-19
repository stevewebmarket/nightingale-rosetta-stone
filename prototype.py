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

# Constants
SR = 22050
HOP_LENGTH = 512
N_BINS = 384  # Increased for better frequency resolution
MIN_FREQ = 20.0
MAX_FREQ = SR / 2
ONSET_STRENGTH_THRESHOLD = 0.5
COHERENCE_WINDOW = 10  # Window for coherence calculation

# Available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def detect_onsets(y, sr, hop_length):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    return onset_times

def compute_iois(onset_times):
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    return iois, mean_ioi

def compute_rhythm_coherence(iois, window=COHERENCE_WINDOW):
    if len(iois) < window:
        return 0.0
    variances = [np.var(iois[i:i+window]) for i in range(len(iois) - window + 1)]
    coherence = 1 / (np.mean(variances) + 1e-6)  # Inverse variance for coherence, avoid div by zero
    return coherence

def compute_rhythm_lattice(iois, mean_ioi):
    if len(iois) == 0:
        return 0.0, 0.0
    # Improved lattice: use gcd of quantized IOIs for base, with finer quantization
    quantized_iois = np.round(iois / 0.01) * 0.01  # Quantize to 10ms resolution
    gcd = scipy.stats.mode(quantized_iois, keepdims=False).mode
    base = gcd if gcd > 0 else mean_ioi / 4
    # Lattice coherence: fraction of IOIs that are multiples of base
    multiples = np.round(iois / base)
    coherence = np.mean(np.abs(multiples * base - iois) < 0.01)  # Tolerance for matching
    return base, coherence

def compute_cqt(y, sr, hop_length, n_bins, min_freq, max_freq):
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=min_freq, n_bins=n_bins, bins_per_octave=48)  # Increased bins_per_octave for better invariance
    cqt_mag = np.abs(cqt)
    return cqt_mag

def compute_cqt_shift_invariance(cqt, shifts=[1, 2, 3]):  # Check invariance over small shifts
    invariance_scores = []
    for shift in shifts:
        rolled = np.roll(cqt, shift, axis=0)
        # Normalize and compute correlation
        norm_cqt = cqt / (np.linalg.norm(cqt) + 1e-6)
        norm_rolled = rolled / (np.linalg.norm(rolled) + 1e-6)
        correlation = np.dot(norm_cqt.flatten(), norm_rolled.flatten())
        invariance_scores.append(correlation)
    return np.mean(invariance_scores)

def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    centroid_type = compute_spectral_centroid(y, sr)
    onset_times = detect_onsets(y, sr, HOP_LENGTH)
    iois, mean_ioi = compute_iois(onset_times)
    rhythm_coherence = compute_rhythm_coherence(iois)
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois, mean_ioi)
    cqt = compute_cqt(y, sr, HOP_LENGTH, N_BINS, MIN_FREQ, MAX_FREQ)
    invariance_metric = compute_cqt_shift_invariance(cqt)

    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_type} sound.")
    print(f"  Detected onsets: {len(onset_times)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)\n")

if __name__ == "__main__":
    if not WAV_FILES:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example (sine wave)
        duration = 5.0
        sr = SR
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        # Save as temp.wav or process directly
        centroid_type = compute_spectral_centroid(y, sr)
        onset_times = detect_onsets(y, sr, HOP_LENGTH)
        iois, mean_ioi = compute_iois(onset_times)
        rhythm_coherence = compute_rhythm_coherence(iois)
        lattice_base, lattice_coherence = compute_rhythm_lattice(iois, mean_ioi)
        cqt = compute_cqt(y, sr, HOP_LENGTH, N_BINS, MIN_FREQ, MAX_FREQ)
        invariance_metric = compute_cqt_shift_invariance(cqt)
        
        print("Analysis for synthetic sine wave:")
        print(f"  Detected {centroid_type} sound.")
        print(f"  Detected onsets: {len(onset_times)}")
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)\n")
    else:
        for wav in WAV_FILES:
            analyze_audio(wav)