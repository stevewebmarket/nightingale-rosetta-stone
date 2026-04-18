# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Depreciation + Autocorrelation + CQT Enhancements
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    print(f"Analyzing {file}")

    # For CQT invariance: Compute CQT spectrogram and use for onset strength
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12))
    onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(cqt), sr=sr, aggregate=np.median, max_size=5)

    # Tempo estimation with improved coherence using beat_track
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=512)
    print(f"Estimated tempo for {file}: {tempo} BPM")

    # Tempogram for rhythm lattice: Autocorrelation tempogram with broader handling
    hop_length = 512
    win_length = 384  # Adjustable for broader sound types (e.g., non-metric like birdsong)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=win_length)

    # Autocorrelate along time axis for enhanced rhythm lattice coherence
    tempogram_auto = librosa.autocorrelate(tempogram, axis=0)

    # Example improvement: Compute mean autocorrelation for simple metric applicability
    mean_auto = np.mean(tempogram_auto, axis=0)
    print(f"Mean autocorrelation shape for {file}: {mean_auto.shape}")

    # Further analysis can be added here for structure-specific invariance

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)