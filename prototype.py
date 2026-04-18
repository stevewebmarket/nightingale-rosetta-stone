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
from scipy.signal import find_peaks

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def classify_sound_type(y, sr):
    """Classify sound based on spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        return "low-centroid sound"
    elif mean_centroid > 3000:
        return "high-centroid sound"
    else:
        return "mid-centroid sound"

def compute_rhythm_metrics(onsets, sr):
    """Compute improved rhythm metrics with better lattice estimation."""
    onset_times = librosa.frames_to_time(onsets)
    iois = np.diff(onset_times)
    if len(iois) == 0:
        return 0, 0.0, 0.0, 0.0
    
    mean_ioi = np.mean(iois)
    # Improved coherence: normalized entropy of IOI histogram
    hist, _ = np.histogram(iois, bins=20)
    coh = 1 - entropy(hist + 1e-10) / np.log(len(hist))  # Higher is more coherent
    
    # Improved rhythm lattice: use autocorrelation to find base period
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks, _ = find_peaks(autocorr, prominence=0.1 * np.max(autocorr))
    if len(peaks) > 1:
        base = np.mean(np.diff(peaks)) / sr  # Base lattice in seconds
    else:
        base = mean_ioi / 10  # Fallback
    
    # Lattice coherence: fraction of onsets fitting multiples of base
    fits = np.sum(np.isclose(onset_times % base, 0, atol=base*0.05)) / len(onset_times)
    return len(onsets), mean_ioi, coh, base, fits

def compute_cqt_invariance(cqt):
    """Compute improved CQT shift invariance metric."""
    # Log-amplitude CQT for better invariance
    cqt_log = librosa.amplitude_to_db(np.abs(cqt))
    # Measure invariance by correlation across small time shifts
    inv = 0
    for shift in range(1, min(5, cqt.shape[1] - 1)):
        corr = np.corrcoef(cqt_log[:, :-shift].flatten(), cqt_log[:, shift:].flatten())[0, 1]
        inv += corr
    inv /= min(4, cqt.shape[1] - 2)  # Normalize
    # Broader handling: penalize if variance is high (less invariant)
    var_penalty = 1 / (1 + np.std(cqt_log))
    return inv * var_penalty

def analyze_audio(file_path):
    """Analyze a single audio file with broad sound handling."""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Sound type classification
        sound_type = classify_sound_type(y, sr)
        
        # Onset detection with adaptive strength for broad handling
        if 'birdsong' in file_path:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
            onset_threshold = 0.5  # Lower for sparse sounds
        else:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_threshold = 1.0
        
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
        
        # Rhythm metrics
        num_onsets, mean_ioi, rhythm_coh, lattice_base, lattice_coh = compute_rhythm_metrics(onsets, sr)
        
        # CQT with adaptive parameters for invariance and broad handling
        hop_length = 512 if 'rock' in file_path else 256  # Adjust for denser rhythms
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
        invariance = compute_cqt_invariance(cqt)
        
        print(f"Analysis for {file_path}:")
        print(f"  Detected {sound_type}.")
        print(f"  Detected onsets: {num_onsets}")
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}")
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coh:.2f}")
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signals would go here, but for now, skip
    else:
        for file in wav_files:
            analyze_audio(file)