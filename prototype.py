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

# Define available WAV files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to generate synthetic test signal
def generate_synthetic_signal(duration=10, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [440, 554.37, 659.25]  # A4, C#5, E5
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    signal += 0.5 * np.random.randn(len(t))  # Add some noise
    return signal, sr

# Main processing function
def process_audio(y, sr, filename="synthetic"):
    print(f"Processing: {filename}")
    
    # Compute CQT with improved parameters for broader invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=96, bins_per_octave=12, filter_scale=1.0)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    
    # Onset envelope for rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512, aggregate=np.median)
    
    # Tempo estimation using updated function path, with aggregate for coherence
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=512, aggregate=np.median)
    print(f"Estimated tempo: {tempo}")
    
    # Improved beat tracking for rhythm lattice: use dynamic programming for better coherence
    tempo_val = float(tempo)  # Use scalar tempo
    beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512, bpm=tempo_val)[1]
    print(f"Beat frames: {beat_frames[:10]}")  # Print first 10 for brevity
    
    # Synced chroma with beat synchronization for invariance
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.mean)
    print(f"Synced chroma shape: {chroma_sync.shape}")
    
    # MFCC and delta for broad sound handling
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
    delta_mfcc = librosa.feature.delta(mfcc)
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Delta MFCC shape: {delta_mfcc.shape}")
    
    # Log CQT for output
    print(f"Log CQT shape: {log_cqt.shape}")
    
    # Additional improvements: Compute rhythm lattice coherence metric (placeholder)
    # For demonstration: simple autocorrelation for rhythm coherence
    autocorr = librosa.autocorrelate(onset_env)
    coherence = np.mean(autocorr[:len(autocorr)//2])  # Basic coherence measure
    print(f"Rhythm coherence metric: {coherence:.4f}")
    
    print()  # Separator for multiple files

# Check for available files
if available_files and all(os.path.exists(f) for f in available_files):
    for filename in available_files:
        y, sr = librosa.load(filename, sr=22050)
        process_audio(y, sr, filename)
else:
    print("No WAV files available, using synthetic test signal.")
    y, sr = generate_synthetic_signal()
    process_audio(y, sr)