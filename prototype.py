# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
from librosa.feature.rhythm import tempo as estimate_tempo

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, fall back to synthetic signals
if not wav_files:
    print("No WAV files available. Generating synthetic test signals.")
    sr = 22050
    # Synthetic tone
    y_synth1 = librosa.tone(440, sr=sr, duration=3)
    # Synthetic beat
    click = librosa.clicks(times=np.arange(0, 3, 0.5), sr=sr, length=3*sr)
    y_synth2 = click + 0.5 * librosa.tone(220, sr=sr, duration=3)
    # Synthetic noise
    y_synth3 = np.random.randn(3*sr) * 0.1
    analyses = [
        ('synthetic_tone', y_synth1),
        ('synthetic_beat', y_synth2),
        ('synthetic_noise', y_synth3)
    ]
else:
    print("Analyzing available WAV files.")
    analyses = []
    for file in wav_files:
        y, sr = librosa.load(file, sr=22050)
        analyses.append((file, y))

for name, y in analyses:
    print(f"Analyzing {name}")
    # Separate harmonic and percussive components for better handling of broad sounds
    y_harm, y_perc = librosa.effects.hpss(y)
    
    # Onset envelope from percussive component
    oenv = librosa.onset.onset_strength(y=y_perc, sr=sr)
    
    # Estimated tempo using median for robustness
    tempo = estimate_tempo(onset_envelope=oenv, sr=sr, aggregate=np.median)
    print(f"Estimated tempo for {name}: {tempo} BPM")
    
    # Autocorrelation of onset envelope
    autocorr = librosa.autocorrelate(oenv)
    mean_autocorr = np.mean(autocorr)
    print(f"Mean autocorrelation value for {name}: {mean_autocorr}")
    
    # Chroma using CQT on harmonic component for improved invariance
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    print(f"Chroma shape for {name} (CQT invariant): {chroma.shape}")
    
    # Average RMS (normalized by max RMS for consistency)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms) / (np.max(rms) + 1e-10)  # Avoid division by zero
    print(f"Average RMS for {name} (normalized): {avg_rms}")
    
    # Improved rhythm lattice using tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    tg_mean = np.mean(tempogram, axis=1)
    tempos = librosa.tempo_frequencies(len(tg_mean), sr=sr)
    dominant_idx = np.argmax(tg_mean)
    dominant_tempo = tempos[dominant_idx]
    print(f"Dominant tempo from rhythm lattice for {name}: {dominant_tempo} BPM")
    
    # Improved feature coherence: correlation between onset strength and chroma energy
    chroma_energy = np.mean(chroma, axis=0)
    min_len = min(len(oenv), len(chroma_energy))
    coherence = np.corrcoef(oenv[:min_len], chroma_energy[:min_len])[0, 1]
    print(f"Feature coherence metric for {name}: {coherence}")