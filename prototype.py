# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

# Configuration parameters
sr = 22050
hop_length = 512

# Available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def analyze_audio(file_path, sr, hop_length):
    # Load audio
    y, _ = librosa.load(file_path, sr=sr)
    
    # Compute onset envelope for rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Estimate tempo using corrected function (improved rhythm handling)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    
    # Improved rhythm lattice: Compute tempogram for multi-scale rhythm representation
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    # Enhance coherence by averaging tempogram across time for a lattice summary
    rhythm_lattice = np.mean(tempogram, axis=1)  # Average over time for coherent rhythm profile
    
    # Compute CQT for frequency analysis
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12)
    
    # Improve CQT invariance: Convert to log-amplitude for amplitude invariance
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    
    # Further invariance: Fold to chroma for octave invariance
    chroma = librosa.feature.chroma_cqt(C=np.abs(cqt), sr=sr, hop_length=hop_length)
    # Enhance coherence by syncing chroma with onset envelope (resample to match lengths)
    if chroma.shape[1] != len(onset_env):
        chroma = librosa.util.sync(chroma, chroma.shape[1], len(onset_env))
    coherent_chroma = np.mean(chroma * onset_env[np.newaxis, :], axis=1)  # Weighted average for coherence
    
    # Broad sound handling: Normalize features for diverse inputs (e.g., birdsong vs. music)
    normalized_rhythm = rhythm_lattice / np.max(rhythm_lattice + 1e-6)  # Normalize lattice
    normalized_chroma = coherent_chroma / np.max(coherent_chroma + 1e-6)  # Normalize chroma
    
    # Print summary for each file
    print(f"Analysis for {file_path}:")
    print(f"  Estimated Tempo: {tempo}")
    print(f"  Rhythm Lattice Summary (first 5 values): {normalized_rhythm[:5]}")
    print(f"  Coherent Chroma Profile: {normalized_chroma}")
    print("")

# Analyze each file
for file in files:
    analyze_audio(file, sr, hop_length)