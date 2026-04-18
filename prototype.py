# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice, Enhanced Coherence, Higher-Res CQT, Broad Handling
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    cv = np.std(iois) / np.mean(iois)
    return 1 / (1 + cv)

def compute_lattice_base(iois):
    if len(iois) == 0:
        return 0.001
    ioi_ms = (iois * 1000).astype(int)
    if np.any(ioi_ms == 0):
        return 0.001
    gcd_val = reduce(gcd, ioi_ms)
    return gcd_val / 1000.0

def compute_lattice_coherence(iois, base):
    if base == 0 or len(iois) == 0:
        return 0.0
    errors = iois % base
    relative_errors = errors / base
    coherence = 1 - np.mean(relative_errors)
    return max(0, coherence)

def compute_cqt_invariance(cqt):
    mag = np.abs(cqt)
    shifted = np.roll(mag, 1, axis=0)
    diff = np.mean(np.abs(mag - shifted))
    return diff

print("Analyzing available WAV files.")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    if centroid > 4000:
        print("  Detected high-centroid sound.")
        print("  Using high-sensitivity onset params.")
        onset_params = {'pre_max': 0.01, 'post_max': 0.01, 'wait': 0.01, 'delta': 0.05}
        use_beat = False
    elif centroid > 1500:
        print("  Detected mid-centroid sound.")
        print("  Using mid-sensitivity onset params.")
        onset_params = {'pre_max': 0.03, 'post_max': 0.03, 'wait': 0.03, 'delta': 0.1}
        use_beat = False
    else:
        print("  Detected low-centroid sound.")
        print("  Using low-sensitivity onset params.")
        onset_params = {'pre_max': 0.05, 'post_max': 0.05, 'wait': 0.05, 'delta': 0.2}
        use_beat = True
    
    if use_beat:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        onset_frames = beats
    else:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, **onset_params)
    
    print(f"  Detected onsets: {len(onset_frames)}")
    
    times = librosa.frames_to_time(onset_frames, sr=sr)
    iois = np.diff(times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois):.2f}")
    
    base = compute_lattice_base(iois)
    print(f"  Rhythm lattice base: {base:.3f} s")
    
    lattice_coh = compute_lattice_coherence(iois, base)
    print(f"  lattice coherence: {lattice_coh:.2f}")
    
    cqt = librosa.cqt(y=y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=420, bins_per_octave=60)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    inv_metric = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {inv_metric:.2f} (lower is more invariant)")