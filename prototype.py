# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

def safe_corrcoef(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

# List available WAV files
wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]

if not wav_files:
    print("No WAV files found. Using synthetic test signals.")
    # Synthetic test signal example
    sr = 22050
    t = np.linspace(0, 5, 5 * sr, endpoint=False)
    y = np.sin(440 * 2 * np.pi * t)  # Simple tone
    file = "synthetic_tone"
else:
    print("Analyzing available WAV files.")
    for file in wav_files:
        print(f"Analyzing {file}")
        y, sr = librosa.load(file, sr=22050)

        # Estimated tempo
        tempo = librosa.beat.tempo(y=y, sr=sr)
        print(f"Estimated tempo for {file}: {tempo} BPM")

        # Mean autocorrelation value
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        ac = librosa.autocorrelate(onset_env)
        mean_ac = np.mean(ac)
        print(f"Mean autocorrelation value for {file}: {mean_ac}")

        # Chroma with CQT for invariance
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        print(f"Chroma shape for {file} (CQT invariant): {chroma.shape}")

        # Average RMS (normalized)
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms) / (np.max(rms) + 1e-10)  # Avoid div by zero
        print(f"Average RMS for {file} (normalized): {avg_rms}")

        # Improved rhythm lattice using tempogram for better coherence and broad sound handling
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        mean_tg = np.mean(tempogram, axis=1)
        tempos = librosa.tempo_frequencies(len(mean_tg), sr=sr)
        dominant_idx = np.argmax(mean_tg)
        dominant_tempo = tempos[dominant_idx]
        print(f"Dominant tempo from rhythm lattice for {file}: {dominant_tempo} BPM")

        # Feature coherence metric with safe correlation for invariance and to avoid NaN
        mean_chroma = np.mean(chroma, axis=0)
        coherence = safe_corrcoef(mean_chroma, onset_env)
        print(f"Feature coherence metric for {file}: {coherence}")