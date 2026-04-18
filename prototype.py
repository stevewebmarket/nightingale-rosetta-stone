# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed rhythm lattice bug + improved invariance
# =============================================================================

import numpy as np
import librosa

def rhythm_lattice(beats, tempo, sr, y_length):
    # Ensure tempo is a scalar
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item() if tempo.size == 1 else tempo[0]
    beat_interval = 60.0 / tempo
    
    # Calculate duration correctly
    duration = y_length / sr
    
    # Generate sub-beats at quarter intervals for finer lattice
    sub_beats = np.arange(0, duration, beat_interval / 4)
    
    # Improved rhythm lattice: interpolate for coherence
    # Create a lattice signal aligned with beats and sub-beats
    times = np.linspace(0, duration, y_length)
    lattice = np.zeros(y_length)
    
    # Mark beats and sub-beats with decaying impulses for coherence
    for t in np.concatenate((beats, sub_beats)):
        if t < duration:
            idx = int(t * sr)
            lattice[idx] = 1.0
            # Add decay for better temporal coherence
            decay_len = int(sr * 0.05)  # 50ms decay
            lattice[idx:idx+decay_len] += np.exp(-np.arange(decay_len) / (decay_len / 3))
    
    # Normalize for invariance
    lattice = lattice / np.max(lattice) if np.max(lattice) > 0 else lattice
    
    return lattice

def main():
    sr = 22050
    # Since no WAV files available, generate synthetic test signal
    # Synthetic signal: 440Hz tone with modulated amplitude to simulate beats
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add synthetic beats: amplitude modulation at ~120 BPM
    beat_freq = 120 / 60.0  # Hz
    y *= (1 + 0.5 * np.sin(2 * np.pi * beat_freq * t))
    # Add some noise for broad sound handling
    y += 0.1 * np.random.randn(len(y))
    y = y.astype(np.float32)
    
    # Compute CQT with improvements for invariance (normalize)
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
    cqt_mag = np.abs(cqt)
    # Normalize for scale invariance
    cqt_mag = cqt_mag / np.max(cqt_mag) if np.max(cqt_mag) > 0 else cqt_mag
    print("CQT shape:", cqt.shape)
    
    # Onset strength for tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    
    # Estimate tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    print("Estimated tempo:", tempo)
    
    # Convert beat frames to times
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    
    y_length = len(y)
    
    # Generate rhythm lattice with improvements
    y_beat = rhythm_lattice(beats, tempo, sr, y_length)
    
    # For demonstration, print some stats
    print("Rhythm lattice generated with length:", len(y_beat))
    print("Max lattice value:", np.max(y_beat))

if __name__ == "__main__":
    main()