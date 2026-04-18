# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + CQT Invariance + Broad Sound Adaptivity
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

# Constants for analysis
SR = 22050
HOP_LENGTH = 512
N_BINS_PER_OCTAVE = 48  # Increased for finer resolution and better invariance
N_OCTAVES = 8  # Adjusted for broader frequency coverage
CENTROID_LOW_THRESH = 1500  # Hz
CENTROID_HIGH_THRESH = 4000  # Hz

def detect_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < CENTROID_LOW_THRESH:
        return "low-centroid"
    elif mean_centroid > CENTROID_HIGH_THRESH:
        return "high-centroid"
    else:
        return "mid-centroid"

def get_onset_params(sound_type):
    if sound_type == "low-centroid":
        return {'backtrack': True, 'pre_max': 0.05, 'post_max': 0.05, 'delta': 0.2}  # Low sensitivity for bass-heavy
    elif sound_type == "high-centroid":
        return {'backtrack': True, 'pre_max': 0.02, 'post_max': 0.02, 'delta': 0.05}  # High sensitivity for transients
    else:
        return {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03, 'delta': 0.1}  # Mid sensitivity

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0, 0, 0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
    
    # Improved rhythm lattice: Use autocorrelation for base period
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
    if len(peaks) > 0:
        lattice_base = np.mean(np.diff(peaks)) * mean_ioi / len(peaks)
    else:
        lattice_base = mean_ioi * 1.5  # Fallback
    
    # Lattice coherence: Entropy-based uniformity
    hist, _ = np.histogram(iois, bins=20)
    lattice_coherence = 1 - entropy(hist + 1e-10) / np.log(len(hist))
    
    return len(onset_times), mean_ioi, coherence, lattice_base, lattice_coherence

def compute_cqt_invariance(cqt):
    # Improved CQT: Hybrid CQT for better shift invariance
    cqt_abs = np.abs(cqt)
    
    # Shift invariance metric: Average cosine similarity across small time shifts
    similarities = []
    for shift in range(1, 11):  # Check invariance to shifts of 1-10 frames
        shifted = np.roll(cqt_abs, shift, axis=1)
        sim = np.mean(cosine_similarity(cqt_abs.T, shifted.T).diagonal())
        similarities.append(sim)
    invariance = np.mean(similarities)
    
    return cqt.shape, cqt.shape[0], invariance

def analyze_audio(file):
    y, sr = librosa.load(file, sr=SR)
    
    sound_type = detect_sound_type(y, sr)
    print(f"  Detected {sound_type} sound.")
    
    onset_params = get_onset_params(sound_type)
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    n_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onset_times)
    print(f"  Detected onsets: {n_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Adaptive CQT bins for broad sound handling
    min_freq = 20 if sound_type == "low-centroid" else 50 if sound_type == "mid-centroid" else 100
    cqt = librosa.hybrid_cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=N_BINS_PER_OCTAVE * N_OCTAVES,
                             bins_per_octave=N_BINS_PER_OCTAVE, fmin=min_freq, tuning=0.0)
    cqt_shape, n_bins, invariance = compute_cqt_invariance(cqt)
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic signals
        print("No WAV files available. Using synthetic test signals.")
        duration = 5.0
        sr = SR
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Synthetic birdsong-like: high freq chirps
        y_bird = np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 5 * t) > 0.5)
        analyze_audio_synthetic(y_bird, sr, "synthetic_birdsong")
        
        # Synthetic orchestra-like: harmonics
        y_orch = sum(np.sin(2 * np.pi * f * t) for f in [110, 220, 330, 440])
        analyze_audio_synthetic(y_orch, sr, "synthetic_orchestra")
        
        # Synthetic rock-like: beat with distortion
        y_rock = np.clip(np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t), -1, 1)
        analyze_audio_synthetic(y_rock, sr, "synthetic_rock")
    else:
        for file in files:
            print(f"Analysis for {file}:")
            analyze_audio(file)

def analyze_audio_synthetic(y, sr, name):
    # Similar to analyze_audio but without loading
    sound_type = detect_sound_type(y, sr)
    print(f"  Detected {sound_type} sound for {name}.")
    
    onset_params = get_onset_params(sound_type)
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, **onset_params)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    n_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onset_times)
    print(f"  Detected onsets: {n_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    min_freq = 20 if sound_type == "low-centroid" else 50 if sound_type == "mid-centroid" else 100
    cqt = librosa.hybrid_cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=N_BINS_PER_OCTAVE * N_OCTAVES,
                             bins_per_octave=N_BINS_PER_OCTAVE, fmin=min_freq, tuning=0.0)
    cqt_shape, n_bins, invariance = compute_cqt_invariance(cqt)
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    main()