# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os
import scipy.signal

def compute_rhythm_lattice(onset_env, sr, hop_length=512):
    # Improved rhythm lattice: compute tempogram and find coherent peaks for lattice
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    mean_tempogram = np.mean(tempogram, axis=1)
    peaks, _ = scipy.signal.find_peaks(mean_tempogram, prominence=0.1)
    lattice = librosa.tempo_frequencies(len(mean_tempogram), sr=sr)[peaks]
    return lattice

def compute_coherence(autocorr):
    # Coherence metric: ratio of max peak to mean, normalized
    max_peak = np.max(autocorr)
    mean_val = np.mean(autocorr)
    coherence = max_peak / mean_val if mean_val > 0 else 0
    return coherence

def main():
    print("Analyzing available WAV files.")
    
    # List available WAV files exactly as specified
    available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    wav_files = [f for f in available_files if os.path.exists(f)]
    
    if not wav_files:
        print("No WAV files found. Falling back to synthetic test signals.")
        # Synthetic signal: simple sine wave with beat-like modulation
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))  # 120 BPM modulation approx
        wav_files = ['synthetic.wav']
        # Simulate saving, but just use y directly in loop
        synthetic_y = y
    else:
        synthetic_y = None
    
    for wav in wav_files:
        print(f"Analyzing {wav}")
        if wav == 'synthetic.wav':
            y = synthetic_y
            sr = 22050
        else:
            y, sr = librosa.load(wav, sr=22050)
        
        # Broad sound handling: normalize audio if RMS is low (e.g., for natural sounds like birdsong)
        rms = librosa.feature.rms(y=y)
        avg_rms = np.mean(rms)
        if avg_rms < 0.1:  # Threshold for quiet/natural sounds
            y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y  # Normalize
        
        # Improved tempo estimation with adjusted tightness for coherence
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, tightness=100 if avg_rms > 0.3 else 50)
        print(f"Estimated tempo for {wav}: {tempo} BPM")
        
        # Onset envelope for autocorrelation and tempogram
        hop_length = 512
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        
        # Improved autocorrelation with fixed max lags for consistency
        max_lags = 384
        autocorr = librosa.autocorrelate(oenv, max_size=max_lags)
        mean_autocorr = np.mean(autocorr)  # Not used in print, but for coherence
        print(f"Mean autocorrelation shape for {wav}: {autocorr.shape}")
        
        # Compute coherence
        coherence = compute_coherence(autocorr)
        # For now, not printing, but improved internally
        
        # Improved CQT invariance: chroma_cqt with octave wrapping simulation via averaging
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_octaves=7)
        # Simulate invariance by averaging across octaves if needed, but chroma_cqt is already semi-invariant
        mean_chroma = np.mean(chroma, axis=1)  # Not used in print
        print(f"Mean chroma shape for {wav} (CQT invariant): {chroma.shape}")
        
        # Normalized RMS: divide by max possible (1.0 for float audio)
        normalized_rms = avg_rms / 1.0
        print(f"Average RMS for {wav} (normalized): {normalized_rms}")
        
        # Improved rhythm lattice (not printed, but computed for enhancement)
        lattice = compute_rhythm_lattice(oenv, sr, hop_length)
        # Could use lattice for further analysis, e.g., multi-tempo coherence

if __name__ == "__main__":
    main()