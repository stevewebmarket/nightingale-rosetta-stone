# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def detect_onsets(y, sr):
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    return onset_times

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    coherence = 1 - np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    return mean_ioi, max(0, coherence)

def adaptive_rhythm_lattice(onset_times, mean_ioi):
    if len(onset_times) < 2:
        return 0.001, 1.0
    iois = np.diff(onset_times)
    gcd_ioi = np.gcd.reduce((iois * 1000).astype(int)) / 1000.0
    base = max(gcd_ioi, mean_ioi / 10) if gcd_ioi > 0 else 0.001
    lattice = np.arange(0, onset_times[-1] + base, base)
    hits = np.sum(np.isclose(np.mod(onset_times, base), 0, atol=1e-3))
    coherence = hits / len(onset_times) if len(onset_times) > 0 else 1.0
    return base, coherence

def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=168, bins_per_octave=24)
    return np.abs(cqt)

def cqt_shift_invariance(cqt, shifts=[1, 2, 3]):
    orig = np.mean(cqt, axis=1)
    invariance = []
    for shift in shifts:
        shifted = np.roll(cqt, shift, axis=1)
        shifted_mean = np.mean(shifted, axis=1)
        corr = np.corrcoef(orig, shifted_mean)[0, 1]
        invariance.append(corr)
    return np.mean(invariance)

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    centroid_type = compute_spectral_centroid(y, sr)
    onset_times = detect_onsets(y, sr)
    mean_ioi, rhythm_coherence = compute_rhythm_metrics(onset_times)
    lattice_base, lattice_coherence = adaptive_rhythm_lattice(onset_times, mean_ioi)
    cqt = compute_cqt(y, sr)
    invariance_metric = cqt_shift_invariance(cqt)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {centroid_type} sound.")
    print(f"  Detected onsets: {len(onset_times)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)

if __name__ == "__main__":
    main()