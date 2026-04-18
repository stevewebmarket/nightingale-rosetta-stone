# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Error Fix + Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def compute_rhythm_lattice(onset_times):
    """Improved rhythm lattice: cluster intervals for better coherence."""
    intervals = np.diff(onset_times)
    # Quantize to a lattice (e.g., multiples of 0.1s for simplicity)
    lattice = np.round(intervals / 0.1) * 0.1
    coherence = 1 - np.var(lattice) / np.mean(lattice) if np.mean(lattice) > 0 else 0
    return lattice, coherence

def analyze(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Compute spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    # Classify sound type and adjust onset detection
    if mean_centroid < 1500:
        sound_type = "low-centroid sound (e.g., ambient)"
        onset_params = {'backtrack': True, 'tightness': 50}  # Loose for broad, ambient sounds
    elif mean_centroid > 4000:
        sound_type = "high-centroid sound (e.g., percussive)"
        onset_params = {'backtrack': False, 'tightness': 200}  # Tight for sharp sounds
    else:
        sound_type = "mid-centroid sound (e.g., orchestral)"
        onset_params = {}  # Balanced defaults
    
    print(f"  Detected {sound_type}, using {'loose' if 'low' in sound_type else 'tight' if 'high' in sound_type else 'balanced'} onset detection.")
    
    # Detect onsets with adjusted parameters
    onsets = librosa.onset.onset_detect(y=y, sr=sr, **onset_params)
    print(f"  Onsets detected: {len(onsets)}")
    
    # Compute onset times
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    
    # Improved rhythm lattice and coherence
    lattice, coherence = compute_rhythm_lattice(onset_times)
    print(f"  Rhythm lattice coherence: {coherence:.2f}")
    
    # Tempo estimation with aggregate to avoid array format issues
    oe = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=oe, sr=sr, aggregate=np.mean)
    print(f"  Estimated tempo: {tempo:.2f} BPM")
    
    # CQT with invariance improvements (log amplitude for scale invariance)
    # Adjust fmin for broad sound handling based on type
    fmin = librosa.note_to_hz('C2') if 'low' in sound_type else librosa.note_to_hz('C1') if 'high' in sound_type else None
    cqt = librosa.cqt(y=y, sr=sr, fmin=fmin)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt))  # Log scale for invariance
    print(f"  CQT shape (invariant): {cqt_db.shape}")
    
    # Additional broad sound handling: adjust hop_length dynamically
    hop_length = 512 if 'mid' in sound_type else 1024 if 'low' in sound_type else 256
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
    print(f"  MFCC mean (adjusted hop): {np.mean(mfcc):.2f}")

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        print(f"Analysis for {file}:")
        try:
            analyze(file)
        except Exception as e:
            print(f"  Error analyzing {file}: {str(e)}")
        print("---")