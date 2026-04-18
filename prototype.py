# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def analyze_audio(y, sr, hop_length, file_name):
    print(f"Analyzing {file_name}")
    tempo = librosa.beat.tempo(y=y, sr=sr, hop_length=hop_length)
    print(f"Estimated tempo for {file_name}: {tempo} BPM")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    ac = librosa.autocorrelate(onset_env)
    mean_autocorr = np.mean(ac)
    print(f"Mean autocorrelation value for {file_name}: {mean_autocorr}")

    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    print(f"Chroma shape for {file_name} (CQT invariant): {chroma.shape}")

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    avg_rms = np.mean(rms)
    print(f"Average RMS for {file_name} (normalized): {avg_rms}")

    # Improved rhythm lattice: ignore zero lag to avoid inf, find max lag
    ac[0] = 0
    dominant_lag_idx = np.argmax(ac)
    if dominant_lag_idx == 0:
        dom_bpm = float('inf')
    else:
        beat_duration = dominant_lag_idx * hop_length / sr
        dom_bpm = 60 / beat_duration
    print(f"Dominant tempo from rhythm lattice for {file_name}: {dom_bpm} BPM")

    # Improved feature coherence: average correlation per pitch class
    T = min(chroma.shape[1], len(onset_env))
    corrs = [np.corrcoef(chroma[i, :T], onset_env[:T])[0, 1] for i in range(12)]
    coherence = np.mean(corrs)
    print(f"Feature coherence metric for {file_name}: {coherence}")

if __name__ == "__main__":
    sr = 22050
    hop_length = 512
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    available = [f for f in files if os.path.exists(f)]

    if available:
        print("Analyzing available WAV files.")
        for file in available:
            y, _ = librosa.load(file, sr=sr)
            analyze_audio(y, sr, hop_length, file)
    else:
        print("No WAV files available. Analyzing synthetic test signals.")
        # Synthetic tone
        y = librosa.tone(440, sr=sr, duration=5)
        analyze_audio(y, sr, hop_length, "synthetic_tone")
        # Synthetic chirp for variety
        y = librosa.chirp(fmin=220, fmax=880, sr=sr, duration=5)
        analyze_audio(y, sr, hop_length, "synthetic_chirp")