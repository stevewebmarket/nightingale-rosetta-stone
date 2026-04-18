# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Tempo Estimation + Rhythm Lattice Improvements
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    hop_length = 512
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Tempo estimation (fixed to use librosa.beat.tempo)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print(f"Estimated tempo for {file}: {tempo} BPM")
    
    # Improved rhythm lattice: Compute tempogram for better rhythm representation
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    # Simple coherence measure: Autocorrelation of tempogram for rhythmic stability
    tempogram_auto = librosa.util.autocorrelate(tempogram, axis=0)
    coherence = np.mean(np.max(tempogram_auto, axis=0))
    print(f"Rhythmic coherence for {file}: {coherence}")
    
    # CQT for spectral features with invariance (log-frequency scaling)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    # Invariance: Normalize for shift-invariance in octaves
    cqt_norm = librosa.util.normalize(cqt_mag, axis=0)
    print(f"CQT shape for {file}: {cqt_norm.shape}")
    
    # Broad sound handling: Additional features for non-musical sounds (e.g., spectral centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    print(f"Average spectral centroid for {file}: {np.mean(centroid)} Hz")

# List of available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Analyze each file
for file in files:
    analyze_audio(file)