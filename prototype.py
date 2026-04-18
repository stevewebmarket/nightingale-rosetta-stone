# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Deprecation Fix, Refined Rhythm Lattice, CQT Normalization, Adaptive Enhancements
# =============================================================================

import librosa
import numpy as np

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    y, sr = librosa.load(file, sr=22050)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Improved tempo estimation with aggregate median for robustness
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    
    print(f"Processing: {file}")
    print(f"Estimated tempo: {tempo}")
    
    tempo_val = tempo[0]
    
    # Improved beat tracking with tempo hint for better coherence
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, bpm=tempo_val, tight=True)
    print(f"Number of beats detected: {len(beats)}")
    
    # Improved rhythm lattice: use regular grid based on mean beat interval for better invariance and coherence
    if len(beats) > 1:
        mean_interval = np.mean(np.diff(beats))
        grid = np.arange(beats[0], beats[-1] + mean_interval, mean_interval)
        grid_points = len(grid)
    else:
        grid_points = len(beats)
    print(f"Rhythm lattice grid points: {grid_points}")
    
    # CQT with normalization for improved invariance to amplitude variations
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length)
    mag = np.abs(cqt)
    max_mag = np.max(mag) if np.max(mag) > 0 else 1.0
    mag_norm = mag / max_mag
    print(f"CQT shape: {mag_norm.shape}")
    
    # Improved coherence score: sum of onset strengths at beats, normalized by number of beats for comparability
    coherence_score = np.sum(onset_env[beats]) / max(1, len(beats))
    print(f"Coherence score: {coherence_score}")
    
    # Improved adaptive mode for broader sound handling: check energy and rhythm stability
    rms = librosa.feature.rms(y=y)
    mean_rms = np.mean(rms)
    energy_level = "High" if mean_rms > 0.1 else "Low"
    
    if len(beats) > 1:
        ibi = np.diff(beats)
        rhythm_stability = "Steady" if np.std(ibi) / np.mean(ibi) < 0.1 else "Variable"
    else:
        rhythm_stability = "Undefined"
    
    adaptive_mode = f"{energy_level} energy sound detected. Rhythm: {rhythm_stability}."
    print(f"Adaptive mode: {adaptive_mode}")
    print("Analysis complete.\n")