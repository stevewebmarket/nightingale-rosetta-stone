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

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Parameters
sr = 22050
hop_length = 512
n_fft = 2048

def analyze_audio(file_path):
    print(f"Analyzing {file_path}")
    y, sr_loaded = librosa.load(file_path, sr=sr)
    
    # Estimated tempo using updated function path
    from librosa.feature import rhythm
    tempo = rhythm.tempo(y=y, sr=sr, hop_length=hop_length)
    print(f"Estimated tempo for {file_path}: {tempo} BPM")
    
    # Onset envelope for autocorrelation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
    autocorrelation = librosa.autocorrelate(onset_env)
    mean_autocorr = np.mean(autocorrelation)
    print(f"Mean autocorrelation value for {file_path}: {mean_autocorr}")
    
    # Chroma with CQT for invariance (improved with more octaves for broad handling)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_octaves=7, bins_per_octave=36)
    print(f"Chroma shape for {file_path} (CQT invariant): {chroma.shape}")
    
    # Average RMS normalized
    rms = librosa.feature.rms(y=y)[0]
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
    avg_rms = np.mean(rms_normalized)
    print(f"Average RMS for {file_path} (normalized): {avg_rms}")
    
    # Improved rhythm lattice: Use tempogram for better rhythm representation and dominant tempo extraction
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    # Compute auto-correlation of tempogram for lattice coherence
    tempogram_ac = np.mean([librosa.autocorrelate(tempogram[i]) for i in range(tempogram.shape[0])], axis=0)
    # Find dominant lag and convert to BPM (improved peak detection with prominence)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(tempogram_ac, prominence=0.1 * np.max(tempogram_ac))
    if len(peaks) > 0:
        dominant_lag = peaks[0]  # First prominent peak
        dominant_period = dominant_lag * hop_length / sr
        dominant_tempo = 60 / dominant_period if dominant_period > 0 else 0
    else:
        dominant_tempo = 0
    print(f"Dominant tempo from rhythm lattice for {file_path}: {dominant_tempo} BPM")
    
    # Improved feature coherence metric: Correlation between chroma and tempogram (enhanced alignment)
    min_len = min(chroma.shape[1], tempogram.shape[1])
    chroma_resized = chroma[:, :min_len]
    tempogram_resized = tempogram[:, :min_len]
    # Mean correlation across dimensions
    coherence = np.mean([np.corrcoef(chroma_resized[i], tempogram_resized[j])[0,1] 
                         for i in range(chroma_resized.shape[0]) 
                         for j in range(tempogram_resized.shape[0])])
    print(f"Feature coherence metric for {file_path}: {coherence}")

print("Analyzing available WAV files.")
for wav in wav_files:
    if os.path.exists(wav):
        analyze_audio(wav)
    else:
        print(f"File {wav} not found.")