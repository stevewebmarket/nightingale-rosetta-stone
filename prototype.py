# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def compute_rhythm_lattice(iois):
    if len(iois) == 0:
        return 0.0, 0.0
    iois_ms = (iois * 1000).round().astype(int)
    iois_ms = iois_ms[iois_ms > 0]  # Avoid zero
    if len(iois_ms) == 0:
        return 0.001, 1.0
    lattice_base_ms = reduce(gcd, iois_ms)
    lattice_base = lattice_base_ms / 1000.0
    # Coherence: fraction of IOIs that are approximate multiples
    multiples = [abs(ioi / lattice_base - round(ioi / lattice_base)) < 0.01 for ioi in iois]
    coherence = np.mean(multiples)
    return lattice_base, coherence

def correlation(a, b):
    a_flat = a.flatten() - np.mean(a)
    b_flat = b.flatten() - np.mean(b)
    num = np.sum(a_flat * b_flat)
    den = np.sqrt(np.sum(a_flat**2) * np.sum(b_flat**2)) + 1e-10
    return num / den

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 4000:
        sound_type = 'high-centroid'
    elif centroid < 1000:
        sound_type = 'low-centroid'
    else:
        sound_type = 'mid-centroid'
    
    # Adaptive onset detection: more sensitive for high-centroid sounds
    delta = 0.02 if 'high' in sound_type else 0.07
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, delta=delta)
    num_onsets = len(onset_times)
    
    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv)
    else:
        mean_ioi = 0
        rhythm_coherence = 0
    
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois if num_onsets > 1 else np.array([]))
    
    # CQT with higher resolution for broad handling
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    cqt_shape = cqt.shape
    
    # Improved CQT shift invariance: correlation with pitch-shifted version
    bins_per_semitone = 24 / 12  # 2
    shift_bins = int(bins_per_semitone)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    cqt_shifted = librosa.cqt(y_shifted, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    cqt_mag = np.abs(cqt)
    cqt_shifted_mag = np.abs(cqt_shifted)
    # Adjust shapes if slight mismatch
    min_frames = min(cqt_mag.shape[1], cqt_shifted_mag.shape[1])
    cqt_mag = cqt_mag[:, :min_frames]
    cqt_shifted_mag = cqt_shifted_mag[:, :min_frames]
    rolled_cqt = np.roll(cqt_mag, shift_bins, axis=0)
    invariance_metric = correlation(rolled_cqt, cqt_shifted_mag)
    
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)

if __name__ == '__main__':
    main()