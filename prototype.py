# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 4000:
        return "high"
    elif mean_centroid > 1500:
        return "mid"
    else:
        return "low"

def detect_onsets(y, sr, centroid_type):
    if centroid_type == "high":
        hop_length = 256
        backtrack = True
        pre_post_max = 5
    elif centroid_type == "mid":
        hop_length = 512
        backtrack = True
        pre_post_max = 3
    else:
        hop_length = 1024
        backtrack = False
        pre_post_max = 2
    
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop_length, backtrack=backtrack, pre_max=pre_post_max, post_max=pre_post_max)
    return onsets

def compute_ioi_stats(onset_times):
    if len(onset_times) < 2:
        return 0, 0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
    return mean_ioi, max(0, coherence)

def compute_rhythm_lattice(onset_times, sr):
    if len(onset_times) < 3:
        return 0, 0
    
    iois = np.diff(onset_times)
    # Improved lattice: use autocorrelation for periodicity
    autocorr = np.correlate(iois, iois, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=5)
    if len(peaks) > 0:
        base_period = (peaks[0] + 1) * (1 / sr)  # Convert to seconds
    else:
        base_period = np.median(iois)
    
    # Lattice coherence: fit onsets to lattice grid
    grid = np.arange(0, onset_times[-1] + base_period, base_period)
    hits = 0
    for ot in onset_times:
        if np.min(np.abs(ot - grid)) < 0.05:  # Tolerance
            hits += 1
    lattice_coherence = hits / len(onset_times) if len(onset_times) > 0 else 0
    
    return base_period, lattice_coherence

def compute_cqt(y, sr):
    # Improved CQT: increased bins per octave for better resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=384, bins_per_octave=48)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    return cqt_mag

def cqt_shift_invariance(cqt):
    # Improved metric: average correlation across multiple shifts
    invariance = 0
    for shift in range(1, 13):  # Semitone shifts
        shifted = np.roll(cqt, shift, axis=0)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        invariance += corr
    return invariance / 12

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=22050)
    centroid_type = compute_spectral_centroid(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {centroid_type}-centroid sound.")
    print(f"  Using {centroid_type}-sensitivity onset params.")
    
    onsets = detect_onsets(y, sr, centroid_type)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    print(f"  Detected onsets: {len(onsets)}")
    
    mean_ioi, rhythm_coherence = compute_ioi_stats(onset_times)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times, sr)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ["birdsong.wav", "orchestra.wav", "rock.wav"]
    for file in files:
        analyze_file(file)