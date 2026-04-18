# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

if not files:
    # Fall back to synthetic test signals if no files
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)  # Simple sine wave
    files = ['synthetic.wav']  # Placeholder, but analyze y directly
    print("No WAV files found. Using synthetic test signal.")
else:
    print("Analyzing available WAV files.")

for file in files:
    print(f"Analyzing {file}")
    if 'synthetic' in file:
        # Use synthetic y from above
        pass
    else:
        y, sr = librosa.load(file, sr=22050)
    
    # Improved: Use CQT for onset strength to enhance CQT invariance
    cqt = librosa.cqt(y=y, sr=sr, hop_length=512)
    S = np.abs(cqt)
    oenv = librosa.onset.onset_strength(S=S, lag=1, max_size=1)
    
    # For better coherence, smooth the onset envelope
    oenv = librosa.util.normalize(oenv)
    
    # Standard tempo estimation
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=512)
    
    # Improved rhythm lattice and broad sound handling: If tempo is 0, fall back to tempogram-based estimation
    # Tempogram provides a lattice of tempo strengths
    if tempo[0] == 0:
        tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=512, norm=None)
        mean_tg = np.mean(tg, axis=1)  # Mean over time for coherence
        bpm = librosa.tempo_frequencies(len(mean_tg), hop_length=512, sr=sr)
        idx = np.argmax(mean_tg)
        tempo = np.array([bpm[idx]])
    
    print(f"Estimated tempo for {file}: {tempo} BPM")
    
    # Compute autocorrelation for consistency with previous outputs
    ac = librosa.autocorrelate(oenv)
    print(f"Mean autocorrelation shape for {file}: {ac.shape}")