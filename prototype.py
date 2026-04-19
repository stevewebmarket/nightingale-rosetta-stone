# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np
import math
from functools import reduce

def gcd_list(numbers):
    if not numbers:
        return 1
    return reduce(math.gcd, numbers)

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    # Compute spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        centroid_type = "high"
        print("  Detected high-centroid sound.")
    elif centroid > 1000:
        centroid_type = "mid"
        print("  Detected mid-centroid sound.")
    else:
        centroid_type = "low"
        print("  Detected low-centroid sound.")
    
    # Detect onsets or beats for better coherence
    if centroid_type == "high":
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
        print(f"  Detected onsets: {len(onset_times)}")
    else:
        _, onset_times = librosa.beat.beat_track(y=y, sr=sr, units='time')
        print(f"  Detected onsets: {len(onset_times)}")
    
    iois = np.diff(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 1 / (1 + cv)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  mean IOI: 0.00 s, rhythm coherence: 1.00")
    
    # Improved rhythm lattice using GCD
    ioi_ms = [int(round(i * 1000)) for i in iois if i > 0.01]
    if ioi_ms:
        base_ms = gcd_list(ioi_ms)
        base = base_ms / 1000.0 if base_ms > 0 else 0.001
    else:
        base = 0.001
    print(f"  Rhythm lattice base: {base:.3f} s")
    
    # Lattice coherence
    if len(iois) > 0:
        deviations = [abs((i % base) / base - 0.5) * 2 for i in iois]  # normalized deviation
        lattice_coherence = 1 - np.mean(deviations)
    else:
        lattice_coherence = 1.0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Improved CQT with higher resolution for invariance
    hop_length = 256
    bins_per_octave = 24
    if centroid_type == "high":
        fmin = librosa.note_to_hz('A3')
        n_bins = 24 * 5  # 120 bins
    elif centroid_type == "mid":
        fmin = librosa.note_to_hz('C1')
        n_bins = 24 * 8  # 192 bins
    else:
        fmin = librosa.note_to_hz('C0')
        n_bins = 24 * 9  # 216 bins
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
    
    # Improved shift invariance metric (average cosine similarity over small shifts)
    abs_cqt = np.abs(cqt)
    inv = 0.0
    num_shifts = 3
    for s in range(1, num_shifts + 1):
        shifted = np.roll(abs_cqt, s, axis=1)
        sims = []
        for i in range(abs_cqt.shape[0]):
            a = abs_cqt[i]
            b = shifted[i]
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            else:
                sim = 1.0
            sims.append(sim)
        inv += np.mean(sims)
    inv /= num_shifts
    print(f"  CQT shift invariance metric: {inv:.2f} (higher is more invariant)")