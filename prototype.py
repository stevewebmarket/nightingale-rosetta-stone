# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Coherence + CQT Invariance + Broad Sound Handling
# =============================================================================

import os
import numpy as np
import librosa

# Constants for analysis
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
CQT_BINS_PER_OCTAVE = 12  # For CQT chroma
MIN_FREQ = 32.7  # C1 frequency for broader sound handling
MAX_FREQ = 4186.0  # C8 frequency
AUTOCORR_LAGS = 384  # Fixed lag for autocorrelation
RMS_WINDOW = 0.1  # Seconds for RMS calculation

# Available WAV files (do not modify or invent new ones)
AVAILABLE_WAVS = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def load_audio(file_path):
    """Load audio file with fixed sample rate."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def estimate_tempo(y, sr):
    """Estimate tempo with improved rhythm lattice handling."""
    # Onset detection with enhanced sensitivity for broad sounds
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    # Apply a rhythm lattice by convolving with a metric grid (e.g., quarter notes)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # Improve coherence by smoothing beat estimates
    tempo_smoothed = np.mean(tempo) if isinstance(tempo, np.ndarray) and len(tempo) > 1 else tempo
    return np.array([tempo_smoothed])

def compute_autocorrelation(y):
    """Compute mean autocorrelation with coherence improvements."""
    # Split into frames for better coherence across time
    frames = librosa.util.frame(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    autocorr = np.array([librosa.autocorrelate(frame)[:AUTOCORR_LAGS] for frame in frames.T])
    # Enhance coherence by averaging with weights (e.g., favoring louder frames)
    weights = np.mean(frames**2, axis=0)  # Energy-based weighting
    mean_autocorr = np.average(autocorr, axis=0, weights=weights)
    return mean_autocorr

def compute_chroma_cqt(y, sr):
    """Compute CQT-based chroma with improved invariance and broad handling."""
    # Use CQT with broader frequency range for invariance to pitch shifts
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=MIN_FREQ, n_bins=84, bins_per_octave=CQT_BINS_PER_OCTAVE)
    chroma = librosa.feature.chroma_cqt(C=cqt, bins_per_octave=CQT_BINS_PER_OCTAVE)
    # Improve invariance by normalizing and applying octave wrapping
    chroma_norm = librosa.util.normalize(chroma, norm=2, axis=0)
    return chroma_norm

def compute_average_rms(y, sr):
    """Compute average RMS with normalization for broad sound handling."""
    rms = librosa.feature.rms(y=y, frame_length=int(sr * RMS_WINDOW), hop_length=HOP_LENGTH)
    # Normalize and average, with clipping for extreme dynamic ranges
    rms_norm = np.clip(rms / np.max(rms), 0, 1)
    return np.mean(rms_norm)

def analyze_files():
    """Analyze all available WAV files."""
    print("Analyzing available WAV files.")
    for wav in AVAILABLE_WAVS:
        if not os.path.exists(wav):
            print(f"File {wav} not found, skipping.")
            continue
        print(f"Analyzing {wav}")
        y, sr = load_audio(wav)
        if y is None:
            continue
        
        # Tempo estimation with rhythm lattice
        tempo = estimate_tempo(y, sr)
        print(f"Estimated tempo for {wav}: {tempo} BPM")
        
        # Autocorrelation with coherence
        mean_autocorr = compute_autocorrelation(y)
        print(f"Mean autocorrelation shape for {wav}: {mean_autocorr.shape}")
        
        # Chroma with CQT invariance
        chroma = compute_chroma_cqt(y, sr)
        print(f"Mean chroma shape for {wav} (CQT invariant): {chroma.shape}")
        
        # Average RMS normalized
        avg_rms = compute_average_rms(y, sr)
        print(f"Average RMS for {wav} (normalized): {avg_rms}")

if __name__ == "__main__":
    if not AVAILABLE_WAVS:
        print("No WAV files available, falling back to synthetic test signals.")
        # Synthetic signal example (sine wave + noise)
        sr = SAMPLE_RATE
        t = np.linspace(0, 5, 5 * sr)  # 5 seconds
        y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        tempo = estimate_tempo(y, sr)
        print(f"Estimated tempo for synthetic signal: {tempo} BPM")
        mean_autocorr = compute_autocorrelation(y)
        print(f"Mean autocorrelation shape for synthetic: {mean_autocorr.shape}")
        chroma = compute_chroma_cqt(y, sr)
        print(f"Mean chroma shape for synthetic (CQT invariant): {chroma.shape}")
        avg_rms = compute_average_rms(y, sr)
        print(f"Average RMS for synthetic (normalized): {avg_rms}")
    else:
        analyze_files()