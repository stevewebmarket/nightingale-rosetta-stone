# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os
import scipy.signal

def compute_rhythm_lattice(onset_env, sr, hop_length=512):
    """Improved rhythm lattice with better peak detection and fallback."""
    lags = np.arange(1, len(onset_env))
    autocorr = np.correlate(onset_env, onset_env, mode='full')[len(onset_env):]
    autocorr = autocorr[:min(5000, len(autocorr))]  # Limit for efficiency
    
    # Smooth autocorrelation for better peak finding
    autocorr = scipy.signal.savgol_filter(autocorr, window_length=51, polyorder=3)
    
    # Find peaks with improved parameters
    peaks, _ = scipy.signal.find_peaks(autocorr, height=np.mean(autocorr) * 1.5, distance=10)
    
    if len(peaks) > 0:
        # Select top peaks and compute tempos
        peak_lags = lags[peaks[:5]]  # Top 5 peaks
        tempos = 60 * sr / (peak_lags * hop_length)
        dominant_tempo = np.median(tempos)  # Use median for robustness
    else:
        # Fallback to overall tempo estimate if no peaks
        dominant_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    return dominant_tempo

def compute_feature_coherence(chroma, onset_env):
    """Improved coherence metric using cross-correlation between chroma and onset."""
    chroma_mean = np.mean(chroma, axis=0)
    if len(chroma_mean) != len(onset_env):
        min_len = min(len(chroma_mean), len(onset_env))
        chroma_mean = chroma_mean[:min_len]
        onset_env = onset_env[:min_len]
    
    # Normalize
    chroma_norm = (chroma_mean - np.mean(chroma_mean)) / (np.std(chroma_mean) + 1e-6)
    onset_norm = (onset_env - np.mean(onset_env)) / (np.std(onset_env) + 1e-6)
    
    # Cross-correlation
    corr = np.correlate(chroma_norm, onset_norm, mode='full')
    coherence = np.max(np.abs(corr)) / len(chroma_norm)
    return coherence

def analyze_audio(file_path, sr=22050):
    print(f"Analyzing {file_path}")
    
    y, sr = librosa.load(file_path, sr=sr)
    
    # Estimated tempo
    tempo = librosa.beat.tempo(y=y, sr=sr)
    print(f"Estimated tempo for {file_path}: {tempo} BPM")
    
    # Onset strength for autocorrelation and rhythm
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    autocorr = np.correlate(onset_env, onset_env, mode='full')[len(onset_env):]
    mean_autocorr = np.mean(autocorr)
    print(f"Mean autocorrelation value for {file_path}: {mean_autocorr}")
    
    # Improved CQT chroma with tuning for invariance and broad sounds
    n_octaves = 7 if 'birdsong' in file_path else 8  # Adaptive for broad handling
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_octaves=n_octaves, threshold=0.0)
    print(f"Chroma shape for {file_path} (CQT invariant): {chroma.shape}")
    
    # Average RMS normalized
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms) / np.max(rms + 1e-6)  # Improved normalization
    print(f"Average RMS for {file_path} (normalized): {avg_rms}")
    
    # Dominant tempo from improved rhythm lattice
    dominant_tempo = compute_rhythm_lattice(onset_env, sr, hop_length)
    print(f"Dominant tempo from rhythm lattice for {file_path}: {dominant_tempo} BPM")
    
    # Improved feature coherence
    coherence = compute_feature_coherence(chroma, onset_env)
    print(f"Feature coherence metric for {file_path}: {coherence}")

def main():
    print("Analyzing available WAV files.")
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    
    for file in wav_files:
        if os.path.exists(file):
            analyze_audio(file)
        else:
            print(f"File {file} not found, skipping.")

if __name__ == "__main__":
    main()