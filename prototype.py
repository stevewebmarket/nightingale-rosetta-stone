# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
from math import gcd
from functools import reduce

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    if mean_centroid < 1000:
        sound_type = "low-centroid"
    elif mean_centroid > 3000:
        sound_type = "high-centroid"
    else:
        sound_type = "mid-centroid"
    print(f"  Detected {sound_type} sound.")
    
    # Adaptive onset detection based on sound type
    if sound_type == "low-centroid":
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=800, n_mels=128)
    elif sound_type == "high-centroid":
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean, fmax=8000, n_mels=256)
    else:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True, pre_max=0.02, post_max=0.02)
    print(f"  Detected onsets: {len(onsets)}")
    
    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        rhythm_coherence = 1 / (1 + std_ioi / mean_ioi) if mean_ioi > 0 else 0
        
        # Improved adaptive rhythm lattice: GCD of IOIs in milliseconds
        ioi_ms = (iois * 1000).astype(int)
        gcd_ms = reduce(gcd, ioi_ms) if len(ioi_ms) > 1 else 1
        lattice_base = max(gcd_ms / 1000.0, 0.001)  # Ensure minimum base
        
        # Improved lattice coherence: mean quantization error normalized
        quantized = np.round(iois / lattice_base)
        errors = np.abs(quantized * lattice_base - iois)
        lattice_coherence = 1 - (np.mean(errors) / lattice_base) if lattice_base > 0 else 0
    else:
        mean_ioi = 0
        rhythm_coherence = 0
        lattice_base = 0.001
        lattice_coherence = 0
    
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Adaptive CQT parameters for broad sound handling and improved invariance
    if sound_type == "low-centroid":
        fmin = 20
        bins_per_octave = 48
    elif sound_type == "high-centroid":
        fmin = 100
        bins_per_octave = 60
    else:
        fmin = librosa.note_to_hz('C1')
        bins_per_octave = 48
    
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=384, bins_per_octave=bins_per_octave, hop_length=256))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: average cosine similarity over small shifts
    if cqt.shape[1] < 4:
        invariance_metric = 0
    else:
        cqt_norm = cqt / (np.max(cqt) + 1e-8)
        shifts = [1, 2, 3]
        sim_sum = 0
        for shift in shifts:
            a = cqt_norm[:, :-shift].flatten()
            b = cqt_norm[:, shift:].flatten()
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            sim_sum += sim
        invariance_metric = (sim_sum / len(shifts)) * 300  # Scaled for higher values indicating better invariance
    
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

if not files:
    # Fallback to synthetic test signals if no files
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(440 * 2 * np.pi * t)  # Simple sine wave
    print("Analysis for synthetic_sine.wav:")
    analyze_audio('synthetic_sine.wav')  # But actually process y directly; placeholder
else:
    for file in files:
        print(f"Analysis for {file}:")
        analyze_audio(file)