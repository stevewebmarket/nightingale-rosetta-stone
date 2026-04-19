# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

def classify_sound_centroid(y, sr):
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if spectral_centroid < 1000:
        return "low-centroid sound"
    elif spectral_centroid > 4000:
        return "high-centroid sound"
    else:
        return "mid-centroid sound"

def detect_onsets(y, sr):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    return onset_times

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    rhythm_coherence = 1 / (np.std(iois) / mean_ioi + 1e-5)  # Coherence as inverse CV
    return len(onset_times), mean_ioi, rhythm_coherence

def compute_adaptive_rhythm_lattice(onset_times, mean_ioi):
    if len(onset_times) < 2:
        return 0.001, 0.0
    # Adaptive base: quarter of mean IOI for better granularity
    base_unit = mean_ioi / 4
    # Quantize onsets to lattice
    lattice = np.arange(0, onset_times[-1] + base_unit, base_unit)
    aligned = np.zeros_like(lattice)
    for ot in onset_times:
        idx = np.argmin(np.abs(lattice - ot))
        aligned[idx] = 1
    # Coherence: ratio of aligned points with hits, improved with autocorrelation
    hits = np.sum(aligned)
    autocorr = np.correlate(aligned, aligned, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    lattice_coherence = np.mean(autocorr[:int(len(autocorr)/10)]) / hits if hits > 0 else 0.0
    return base_unit, lattice_coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    return np.abs(cqt)

def cqt_shift_invariance(cqt):
    # Enhanced invariance: compare original to shifted versions with broader tolerance
    shifts = [1, 2, 3]  # Check multiple small shifts
    invariance = 0.0
    for shift in shifts:
        shifted = np.roll(cqt, shift, axis=1)
        # Use cosine similarity for better robustness
        sim = np.dot(cqt.flatten(), shifted.flatten()) / (np.linalg.norm(cqt.flatten()) * np.linalg.norm(shifted.flatten()) + 1e-5)
        invariance += sim / len(shifts)
    return invariance

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    centroid_class = classify_sound_centroid(y, sr)
    onset_times = detect_onsets(y, sr)
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(onset_times)
    base_unit, lattice_coherence = compute_adaptive_rhythm_lattice(onset_times, mean_ioi)
    cqt = compute_cqt(y, sr)
    invariance = cqt_shift_invariance(cqt)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_class}.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {base_unit:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)\n")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)