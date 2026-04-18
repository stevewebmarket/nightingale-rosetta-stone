# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Error Fixes + Enhanced Rhythm Lattice & CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os

def compute_rhythm_lattice(onset_times, tempo):
    # Improved rhythm lattice: build a grid based on tempo and align onsets
    beat_interval = 60 / tempo
    lattice_points = np.arange(onset_times[0], onset_times[-1] + beat_interval, beat_interval)
    # Find deviations
    deviations = np.min([np.abs(ot - lattice_points[:, np.newaxis]) for ot in onset_times], axis=0)
    return lattice_points, deviations

def compute_coherence(deviations):
    # Improved coherence: inverse of normalized variance of deviations
    if len(deviations) == 0:
        return 0.0
    var = np.var(deviations)
    coherence = 1 / (1 + var / 0.1)  # Normalize with a small scale
    return coherence

def analyze_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Compute spectral centroid for classification
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_cent = np.mean(centroid)
        
        # Classify sound type
        if mean_cent > 5000:
            sound_type = "high-centroid sound (e.g., percussive)"
            detection_mode = "tight"
            params = {'pre_max': 1, 'post_max': 1, 'pre_avg': 3, 'post_avg': 3, 'delta': 0.15, 'wait': 1}
            beat_tightness = 100
        elif mean_cent > 2000:
            sound_type = "mid-centroid sound (e.g., orchestral)"
            detection_mode = "balanced"
            params = {'pre_max': 3, 'post_max': 3, 'pre_avg': 5, 'post_avg': 5, 'delta': 0.1, 'wait': 3}
            beat_tightness = 200
        else:
            sound_type = "low-centroid sound (e.g., ambient)"
            detection_mode = "loose"
            params = {'pre_max': 5, 'post_max': 5, 'pre_avg': 7, 'post_avg': 7, 'delta': 0.05, 'wait': 5}
            beat_tightness = 400
        
        print(f"  Detected {sound_type}, using {detection_mode} onset detection.")
        
        # Improved CQT-based onset strength for invariance and broad handling
        cqt = librosa.cqt(y, sr=sr)
        S = librosa.amplitude_to_db(np.abs(cqt))
        oe = librosa.onset.onset_strength(S=S, sr=sr)
        
        # Onset detection with adaptive parameters
        onset_frames = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, **params)
        print(f"  Onsets detected: {len(onset_frames)}")
        
        # Compute tempo using updated function
        tempo = librosa.feature.rhythm.tempo(onset_envelope=oe, sr=sr, aggregate=np.mean)
        
        # Compute beat tracking for lattice alignment
        _, beat_frames = librosa.beat.beat_track(onset_envelope=oe, sr=sr, tightness=beat_tightness)
        
        # Convert to times
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Improved rhythm lattice and coherence
        lattice_points, deviations = compute_rhythm_lattice(onset_times, tempo)
        coherence = compute_coherence(deviations)
        print(f"  Rhythm lattice coherence: {coherence:.2f}")
        
    except Exception as e:
        print(f"  Error analyzing {os.path.basename(file_path)}: {str(e)}")

def main():
    print("Analyzing available WAV files.")
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        if os.path.exists(file):
            print(f"Analysis for {file}:")
            analyze_file(file)
            print("---")
        else:
            print(f"File {file} not found, skipping.")
            print("---")

if __name__ == "__main__":
    main()