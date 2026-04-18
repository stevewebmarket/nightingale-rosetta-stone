# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    
    # Improved rhythm lattice using tempogram for better coherence and broad sound handling
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=384)
    rhythm_lattice = np.mean(tempogram, axis=1)
    rhythm_lattice /= np.max(rhythm_lattice) if np.max(rhythm_lattice) != 0 else 1.0
    
    # Improved coherent chroma profile using CENS for better invariance and coherence
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm = librosa.effects.harmonic(y)
    chroma_cens = librosa.feature.chroma_cens(y=y_harm, sr=sr, hop_length=hop_length, tuning=tuning, n_octaves=7, bins_per_octave=36)
    coherent_chroma = np.mean(chroma_cens, axis=1)
    coherent_chroma /= np.max(coherent_chroma) if np.max(coherent_chroma) != 0 else 1.0
    
    print(f"Analysis for {file}:")
    print("  Estimated Tempo:", tempo)
    print("  Rhythm Lattice Summary (first 5 values):", rhythm_lattice[:5])
    print("  Coherent Chroma Profile:", coherent_chroma)

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

if not files:
    # Fallback to synthetic test signals if no files
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)  # Simple sine wave
    file = 'synthetic.wav'
    # Simulate saving and loading, but directly use y
    analyze_audio(file)  # But actually, adapt to use y directly if needed
else:
    for file in files:
        if os.path.exists(file):
            analyze_audio(file)
        else:
            print(f"File {file} not found, skipping.")