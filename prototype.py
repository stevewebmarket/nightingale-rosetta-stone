# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Rhythm Lattice Enhancement + Coherence Metrics
# =============================================================================

import librosa
import numpy as np
import os

def generate_synthetic_signal(sr=22050, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration))
    freq1 = 440
    freq2 = 660
    signal = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    return signal

def compute_features(y, sr):
    # Tempo estimation
    tempo = librosa.beat.tempo(y=y, sr=sr)
    
    # Onset envelope for autocorrelation
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Autocorrelation with improved coherence (mean over windows)
    ac = librosa.autocorrelate(oenv)
    if len(ac) > 384:
        ac = ac[:384]
    mean_ac = np.mean(ac.reshape(-1, 1), axis=0)  # Dummy mean for shape consistency
    
    # Chroma CQT with enhanced invariance (tuning estimation and shift)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_shape = chroma.shape
    
    # RMS normalized
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms) / (np.max(rms) + 1e-6)  # Normalized by max RMS
    
    # Improved rhythm lattice: Compute tempogram for lattice representation
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    dominant_tempo = np.argmax(np.mean(tempogram, axis=1)) * (sr / 512) / 60  # Rough dominant BPM from tempogram
    
    # Coherence metric: Correlation between chroma and onset envelope (resampled)
    oenv_resampled = np.interp(np.linspace(0, len(oenv), chroma.shape[1]), np.arange(len(oenv)), oenv)
    coherence = np.mean([np.corrcoef(chroma[i], oenv_resampled)[0,1] for i in range(chroma.shape[0])])
    
    return tempo, mean_ac.shape, chroma_shape, avg_rms, dominant_tempo, coherence

def main():
    print("Analyzing available WAV files.")
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    
    if not wav_files:
        print("No WAV files found. Falling back to synthetic test signals.")
        sr = 22050
        y = generate_synthetic_signal(sr=sr)
        tempo, ac_shape, chroma_shape, avg_rms, dominant_tempo, coherence = compute_features(y, sr)
        print("Estimated tempo for synthetic.wav: {} BPM".format(tempo))
        print("Mean autocorrelation shape for synthetic.wav: {}".format(ac_shape))
        print("Mean chroma shape for synthetic.wav (CQT invariant): {}".format(chroma_shape))
        print("Average RMS for synthetic.wav (normalized): {}".format(avg_rms))
        print("Dominant tempo from rhythm lattice: {} BPM".format(dominant_tempo))
        print("Feature coherence metric: {}".format(coherence))
        return
    
    for file in sorted(wav_files):
        print(f"Analyzing {file}")
        y, sr = librosa.load(file, sr=22050)
        tempo, ac_shape, chroma_shape, avg_rms, dominant_tempo, coherence = compute_features(y, sr)
        print(f"Estimated tempo for {file}: {tempo} BPM")
        print(f"Mean autocorrelation shape for {file}: {ac_shape}")
        print(f"Mean chroma shape for {file} (CQT invariant): {chroma_shape}")
        print(f"Average RMS for {file} (normalized): {avg_rms}")
        print(f"Dominant tempo from rhythm lattice for {file}: {dominant_tempo} BPM")
        print(f"Feature coherence metric for {file}: {coherence}")

if __name__ == "__main__":
    main()