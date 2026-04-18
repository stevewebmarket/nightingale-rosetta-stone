# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v17.0 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
from scipy.signal import coherence

print("Analyzing available WAV files.")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f"Analyzing {file}")
    
    y, sr = librosa.load(file, sr=22050)
    
    # Normalize amplitude for broader sound handling (helps with low-energy signals like birdsong)
    y = y / np.max(np.abs(y) + 1e-9)
    
    # Estimated tempo using updated function
    tempo = librosa.feature.tempo(y=y, sr=sr)
    print(f"Estimated tempo for {file}: {tempo} BPM")
    
    # Onset envelope for rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Autocorrelation with improved handling
    autocorr = librosa.autocorrelate(onset_env)
    mean_autocorr = np.mean(autocorr)
    print(f"Mean autocorrelation value for {file}: {mean_autocorr}")
    
    # Chroma with CQT for invariance
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    print(f"Chroma shape for {file} (CQT invariant): {chroma.shape}")
    
    # Average RMS (normalized)
    rms = librosa.feature.rms(y=y)
    avg_rms = np.mean(rms)
    print(f"Average RMS for {file} (normalized): {avg_rms}")
    
    # Improved rhythm lattice: using tempogram to find dominant tempo
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    mean_tg = np.mean(tempogram, axis=1)
    tempos = librosa.tempo_frequencies(len(mean_tg), sr=sr)
    dominant_idx = np.argmax(mean_tg)
    dominant_tempo = tempos[dominant_idx]
    print(f"Dominant tempo from rhythm lattice for {file}: {dominant_tempo} BPM")
    
    # Improved coherence metric: mean coherence between mean chroma over time and onset envelope
    mean_chroma_time = np.mean(chroma, axis=0)
    # Resample if lengths differ (though they should match)
    min_len = min(len(mean_chroma_time), len(onset_env))
    f, Cxy = coherence(mean_chroma_time[:min_len], onset_env[:min_len])
    mean_coherence = np.mean(Cxy)
    print(f"Feature coherence metric for {file}: {mean_coherence}")