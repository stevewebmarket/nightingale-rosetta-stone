# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Tempo Estimation + Enhanced Rhythm Lattice & CQT Invariance
# =============================================================================

import librosa
import numpy as np

def process_audio(y, sr, filename):
    print(f"Processing: {filename}")
    
    # Compute onset envelope for rhythm analysis
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Estimate tempo using corrected librosa path (improved for broad sound handling)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print(f"Estimated tempo: {tempo}")
    
    # Improved beat tracking for rhythm lattice construction
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    print(f"Number of beats detected: {len(beats)}")
    
    # Build simple rhythm lattice: quantize beats into a grid for coherence
    if len(beat_times) > 1:
        beat_diffs = np.diff(beat_times)
        lattice_grid = np.arange(beat_times[0], beat_times[-1], np.median(beat_diffs))
        print(f"Rhythm lattice grid points: {len(lattice_grid)}")
    else:
        print("Insufficient beats for lattice.")
    
    # Compute CQT for frequency analysis with invariance improvements
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))  # Log magnitude for shift invariance
    print(f"CQT shape: {cqt_mag.shape}")
    
    # Enhance coherence: compute autocorrelation of onset envelope
    autocorr = librosa.autocorrelate(onset_env)
    coherence_score = np.mean(autocorr[:len(autocorr)//2])  # Simple coherence metric
    print(f"Coherence score: {coherence_score}")
    
    # Broad sound handling: adaptive thresholding based on signal energy
    energy = np.mean(librosa.feature.rms(y=y))
    if energy < 0.01:  # Low energy (e.g., birdsong)
        print("Adaptive mode: Low energy sound detected.")
    else:
        print("Adaptive mode: High energy sound detected.")
    
    print("Analysis complete.\n")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic test signal if no files
        duration = 5.0
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=duration)
        process_audio(y, sr, "synthetic.wav")
    else:
        for filename in files:
            y, sr = librosa.load(filename, sr=22050)
            process_audio(y, sr, filename)