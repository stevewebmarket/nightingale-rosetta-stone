# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Fallback synthetic signal if no files
if not wav_files:
    print("No WAV files available. Generating synthetic test signal.")
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(440 * 2 * np.pi * t)  # 440 Hz sine wave
    wav_files = ['synthetic']
else:
    print("Analyzing available WAV files.")

for file in wav_files:
    print(f"Analyzing {file}")
    
    if file == 'synthetic':
        # Use the synthetic y and sr
        pass
    else:
        # Load the audio file
        y, sr = librosa.load(file, sr=22050)
    
    # Compute onset envelope for rhythm analysis
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Improved rhythm lattice: Use tempogram for multi-resolution rhythm analysis
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    
    # Estimate tempo from tempogram for coherence
    tempo = librosa.feature.rhythm.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    print(f"Estimated tempo for {file}: {tempo} BPM")
    
    # Enhance coherence: Compute mean autocorrelation across tempogram frames
    autocorr = librosa.autocorrelate(tempogram.T, axis=0)  # Transpose for frame-wise autocorrelation
    mean_autocorr = np.mean(autocorr, axis=0)
    print(f"Mean autocorrelation shape for {file}: {mean_autocorr.shape}")
    
    # CQT invariance: Compute CQT and derive invariant chroma features
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12)
    cqt_mag = np.abs(cqt)
    chroma = librosa.feature.chroma_cqt(C=cqt_mag, sr=sr, hop_length=hop_length, n_chroma=12, threshold=0.0)
    # For demonstration, print mean chroma shape (invariance to octave shifts)
    print(f"Mean chroma shape for {file} (CQT invariant): {chroma.shape}")
    
    # Broad sound handling: Adaptive normalization for diverse sounds (e.g., birdsong noise, orchestral dynamics, rock distortion)
    y_normalized = librosa.util.normalize(y)
    rms = librosa.feature.rms(y=y_normalized)
    print(f"Average RMS for {file} (normalized): {np.mean(rms)}")