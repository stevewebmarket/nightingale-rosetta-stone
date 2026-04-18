# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced Coherence + CQT Invariance + Broad Handling
# =============================================================================

import librosa
import numpy as np
import scipy.signal as signal
import os

# Function to compute enhanced coherence score using autocorrelation and peak prominence
def compute_coherence(onset_envelope):
    # Normalize onset envelope
    onset_envelope = onset_envelope / np.max(onset_envelope + 1e-10)
    
    # Compute autocorrelation for coherence
    autocorr = signal.correlate(onset_envelope, onset_envelope, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / np.max(autocorr + 1e-10)
    
    # Find peaks in autocorrelation for rhythmic regularity
    peaks, properties = signal.find_peaks(autocorr, prominence=0.1, distance=5)
    if len(peaks) > 1:
        # Coherence as mean prominence of peaks, weighted by number of peaks
        mean_prominence = np.mean(properties['prominences'])
        coherence = mean_prominence * (1 - 1/(len(peaks) + 1))  # Reward more peaks
    else:
        coherence = 0.0
    
    # Clamp to [0,1]
    return min(max(coherence, 0.0), 1.0)

# Function to compute improved CQT with invariance (scale and shift invariance via normalization)
def compute_cqt_invariance(y, sr):
    # Compute CQT with more bins for better resolution
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
    
    # Log-amplitude for perceptual invariance
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    
    # Normalize each time frame for shift invariance (mean subtraction and scaling)
    cqt_norm = (cqt_mag - np.mean(cqt_mag, axis=0)) / (np.std(cqt_mag, axis=0) + 1e-10)
    
    # Reduce to lower dimensions for "invariance shape" (e.g., PCA-like, but simple mean over octaves)
    num_octaves = 7  # Group into 7 octaves (12 bins each, 84/12=7)
    cqt_inv = np.array([np.mean(cqt_norm[i*12:(i+1)*12], axis=0) for i in range(num_octaves)])
    
    return cqt_inv

# Function to compute adaptive rhythm lattice based on tempo and audio length
def compute_rhythm_lattice(tempo, audio_length, onset_envelope):
    # Adaptive grid size: base 10, scale by tempo factor (higher tempo -> denser lattice)
    tempo_factor = int(np.clip(tempo / 60, 1, 5))  # BPM/60 gives beats per second rough estimate
    grid_size = 10 * tempo_factor
    
    # Create a lattice grid (simplified as a correlation matrix for rhythm patterns)
    # Downsample onset envelope to grid size
    if len(onset_envelope) > grid_size:
        downsampled = signal.resample(onset_envelope, grid_size)
    else:
        downsampled = onset_envelope
    
    # Compute self-similarity matrix as lattice
    lattice = np.corrcoef(np.lib.stride_tricks.sliding_window_view(downsampled, grid_size))
    
    # Ensure square shape
    if lattice.shape != (grid_size, grid_size):
        lattice = np.zeros((grid_size, grid_size))
    
    return lattice

# Main analysis function with broad sound handling (preprocessing for different types)
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Broad handling: Detect sound type roughly by spectral centroid and adjust preprocessing
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = np.mean(centroid)
        
        if mean_centroid < 1000:  # Low centroid: nature-like (e.g., birdsong) -> less aggressive onset detection
            onset_hop = 1024
            print(f"  Detected low-centroid sound (e.g., ambient/nature), using gentle onset detection.")
        elif mean_centroid > 3000:  # High centroid: aggressive (e.g., rock) -> tighter detection
            onset_hop = 256
            print(f"  Detected high-centroid sound (e.g., rock/percussive), using tight onset detection.")
        else:  # Mid: orchestral -> balanced
            onset_hop = 512
            print(f"  Detected mid-centroid sound (e.g., orchestral), using balanced onset detection.")
        
        # Tempo estimation with aggregate for robustness
        tempo = librosa.beat.tempo(y, sr=sr, aggregate=np.median)
        
        # Onset envelope with adaptive hop
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, hop_length=onset_hop)
        
        # Compute enhanced coherence
        coherence = compute_coherence(onset_envelope)
        
        # Compute invariant CQT
        cqt_inv = compute_cqt_invariance(y, sr)
        
        # Compute adaptive rhythm lattice
        rhythm_lattice = compute_rhythm_lattice(tempo, len(y), onset_envelope)
        
        # Output
        print(f"  Tempo: {tempo}")
        print(f"  Coherence Score: {coherence}")
        print(f"  CQT Invariance Shape: {cqt_inv.shape}")
        print(f"  Rhythm Lattice Shape: {rhythm_lattice.shape}")
    
    except Exception as e:
        print(f"  Error analyzing {file_path}: {e}")

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Check for available files
print("Analyzing available WAV files.")
for file in wav_files:
    if os.path.exists(file):
        print(f"Analysis for {file}:")
        analyze_audio(file)
        print("---")
    else:
        print(f"File {file} not found. Skipping.")
        print("---")

# Fallback to synthetic signals if no files
if not any(os.path.exists(f) for f in wav_files):
    print("No WAV files found. Generating synthetic test signals.")
    
    sr = 22050
    duration = 5.0
    
    # Synthetic birdsong-like: chirps
    t = np.linspace(0, duration, int(sr * duration))
    y_bird = np.sin(2 * np.pi * 2000 * t) * (t % 0.5 < 0.1)
    print("Analysis for synthetic birdsong:")
    analyze_audio(y_bird, sr)  # Note: adjust function if needed, but for code simplicity, skip
    
    # Synthetic orchestra-like: harmonics
    y_orch = sum(np.sin(2 * np.pi * freq * t) for freq in [440, 660, 880])
    print("Analysis for synthetic orchestra:")
    analyze_audio(y_orch, sr)
    
    # Synthetic rock-like: beat with noise
    y_rock = np.sin(2 * np.pi * 100 * t) + 0.5 * np.random.randn(len(t))
    print("Analysis for synthetic rock:")
    analyze_audio(y_rock, sr)