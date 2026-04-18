# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed rhythm lattice boundary; improved coherence and invariance
# =============================================================================

import numpy as np
import librosa

def generate_synthetic_audio(sr=22050, duration=10, tempo=120):
    """Generate a synthetic audio signal with beats for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    beat_interval = 60 / tempo
    beats = np.arange(0, duration, beat_interval)
    y = np.sin(2 * np.pi * 440 * t)  # Base tone
    for beat in beats:
        beat_sample = int(beat * sr)
        y[beat_sample:beat_sample + int(sr * 0.1)] += 0.5 * np.sin(2 * np.pi * 880 * t[:int(sr * 0.1)])
    return y

def rhythm_lattice(beats, tempo, sr, y_length, hop_length=512):
    """Create a rhythm lattice with exponential decay for each beat, improved coherence."""
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]  # Take the first tempo estimate
    beat_duration = 60 / tempo  # seconds per beat
    decay_time = beat_duration / 4  # Decay over quarter beat for finer resolution
    decay_len = int(sr * decay_time / hop_length)  # In frames, for CQT alignment
    
    lattice = np.zeros(y_length)
    for beat in beats:
        idx = int(beat)  # Assuming beats are in frames
        effective_len = min(decay_len, y_length - idx)
        if effective_len > 0:
            decay = np.exp(-np.arange(effective_len) / (effective_len / 3))
            lattice[idx:idx + effective_len] += decay
    
    # Improve coherence: smooth the lattice for better temporal consistency
    lattice = np.convolve(lattice, np.hanning(5), mode='same')
    return lattice / np.max(lattice) if np.max(lattice) > 0 else lattice

def compute_cqt(y, sr, hop_length=512, n_bins=84, bins_per_octave=12):
    """Compute CQT with invariance enhancements."""
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_mag = np.abs(cqt)
    # Invariance: normalize per octave for shift invariance
    for oct in range(7):  # Assuming 7 octaves in 84 bins
        start = oct * 12
        end = start + 12
        if np.max(cqt_mag[start:end]) > 0:
            cqt_mag[start:end] /= np.max(cqt_mag[start:end])
    return cqt_mag

def main():
    sr = 22050
    hop_length = 512
    
    # Since no WAV files, use synthetic
    y = generate_synthetic_audio(sr=sr, tempo=120)
    y_length = len(y)
    
    # Broad sound handling: preprocess for various dynamics
    y = librosa.util.normalize(y)
    
    # Compute CQT with invariance
    cqt = compute_cqt(y, sr, hop_length=hop_length)
    print("CQT shape:", cqt.shape)
    
    # Estimate tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    print("Estimated tempo:", tempo)
    
    # Compute rhythm lattice with improvements
    y_beat = rhythm_lattice(beats, tempo, sr, cqt.shape[1], hop_length=hop_length)  # Align to CQT frames
    
    # For demonstration, combine or analyze further
    # Example: modulate CQT with rhythm lattice for coherence
    modulated = cqt * y_beat[:, np.newaxis].T  # Broadcast rhythm to CQT bins
    print("Modulated shape:", modulated.shape)

if __name__ == "__main__":
    main()