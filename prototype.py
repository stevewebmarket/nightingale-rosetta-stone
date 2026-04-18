# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import librosa
import numpy as np
import scipy.stats

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
    return 1 / (1 + cv)

def find_best_lattice_base(onset_times, min_base=0.05, max_base=0.5, steps=100):
    best_coherence = 0
    best_base = min_base
    for base in np.linspace(min_base, max_base, steps):
        lattice = np.arange(0, onset_times[-1] + base, base)
        coherence = 0
        for ot in onset_times:
            min_dist = min(abs(ot - lt) for lt in lattice)
            coherence += 1 - min(1, min_dist / (base / 2))
        coherence /= len(onset_times)
        if coherence > best_coherence:
            best_coherence = coherence
            best_base = base
    return best_base, best_coherence

def cqt_shift_invariance(cqt, bins_per_octave=48):
    if cqt.shape[0] < bins_per_octave:
        return 0.0
    shifted = np.roll(cqt, bins_per_octave, axis=0)
    corr = np.mean([scipy.stats.pearsonr(cqt[:, i], shifted[:, i])[0] for i in range(cqt.shape[1]) if not np.isnan(scipy.stats.pearsonr(cqt[:, i], shifted[:, i])[0])])
    return max(0, corr)  # Normalize to [0,1], higher better

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    hop_length = 512
    
    # Spectral centroid for sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 5000:
        print(f"  Detected high-centroid sound.")
        print(f"  Using high-sensitivity onset params.")
        delta = 0.05
        wait = 1
    else:
        print(f"  Detected mid-centroid sound.")
        print(f"  Using mid-sensitivity onset params.")
        delta = 0.1
        wait = 2
    
    # Onset detection
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, units='time', backtrack=True, delta=delta, wait=wait)
    
    print(f"  Detected onsets: {len(onset_times)}")
    
    if len(onset_times) < 2:
        print("  Insufficient onsets for rhythm analysis.")
        return
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    rhythm_coh = compute_rhythm_coherence(iois)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}")
    
    # Improved rhythm lattice
    lattice_base, lattice_coh = find_best_lattice_base(onset_times)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coh:.2f}")
    
    # CQT
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT invariance
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic
        duration = 5.0
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(440 * 2 * np.pi * t)  # A4 tone
        # Analyze synthetic, but for now, simulate
        print("No files, using synthetic signal.")
        analyze_audio('synthetic')  # But load is for file, adjust if needed
    else:
        for file in files:
            print(f"Analysis for {file}:")
            analyze_audio(file)