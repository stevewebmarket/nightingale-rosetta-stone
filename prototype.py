# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice Optimization, Coherence, CQT Invariance, and Broad Sound Adaptation
# =============================================================================

import librosa
import numpy as np
import librosa.feature

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
    return max(0.0, 1 - cv)  # Improved: clip to non-negative

def optimize_lattice_base(times):
    if len(times) < 2:
        return 0.0, 0.0
    iois = np.diff(times)
    mean_ioi = np.mean(iois)
    candidates = np.linspace(mean_ioi / 2, mean_ioi * 2, 100)
    best_coh = 0.0
    best_base = 0.0
    for base in candidates:
        phases = np.mod(times, base) / base
        var = np.var(phases)
        coh = max(0.0, 1 - var / (1 / 12))  # Normalized by uniform variance
        if coh > best_coh:
            best_coh = coh
            best_base = base
    return best_base, best_coh

def analyze(file):
    y, sr = librosa.load(file, sr=22050)
    print(f"Analysis for {file}:")
    
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Improved broad sound handling with refined thresholds and params
    if centroid > 6000:
        print("  Detected high-centroid sound.")
        print("  Using high-sensitivity onset params.")
        onset_params = {'backtrack': False, 'pre_max': 0.01, 'post_max': 0.01}
        cqt_params = {'fmin': 500, 'bins_per_octave': 48, 'hop_length': 256}  # Smaller hop for high freq
    elif centroid > 2500:
        print("  Detected mid-centroid sound.")
        print("  Using mid-sensitivity onset params.")
        onset_params = {'backtrack': True, 'pre_max': 0.03, 'post_max': 0.03}
        cqt_params = {'fmin': 100, 'bins_per_octave': 36, 'hop_length': 512}
    else:
        print("  Detected low-centroid sound.")
        print("  Using low-sensitivity onset params.")
        onset_params = {'backtrack': True, 'pre_max': 0.1, 'post_max': 0.1}
        cqt_params = {'fmin': 20, 'bins_per_octave': 24, 'hop_length': 1024}  # Larger hop for low freq
    
    # Onset detection with adaptive params
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, **onset_params)
    print(f"  Detected onsets: {len(onsets)}")
    
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        rhythm_coh = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}")
    else:
        mean_ioi = 0.0
        rhythm_coh = 0.0
    
    # Improved rhythm lattice with optimization
    lattice_base, lattice_coh = optimize_lattice_base(times)
    # Fallback to tempo-based if optimization fails
    if lattice_base == 0.0:
        tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
        lattice_base = 60 / tempo if tempo > 0 else 0.0
        lattice_coh = 0.5  # Default
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coh:.2f}")
    
    # Improved CQT with hybrid for broad handling and adaptive params
    cqt = librosa.hybrid_cqt(y=y, sr=sr, n_bins=384, **cqt_params)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved shift invariance metric: real correlation on magnitude for better invariance measure
    abs_cqt = np.abs(cqt)
    if abs_cqt.shape[1] > 1:
        flat1 = abs_cqt[:, :-1].flatten()
        flat2 = abs_cqt[:, 1:].flatten()
        metric = np.corrcoef(flat1, flat2)[0, 1]
    else:
        metric = 0.0
    print(f"  CQT shift invariance metric: {metric:.2f} (higher is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
for file in files:
    analyze(file)