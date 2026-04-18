# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Rhythm Lattice Enhancements + CQT Invariance Improvements
# =============================================================================

import librosa
import numpy as np

def generate_synthetic_signal(sr=22050, duration=10.0):
    """Generate a synthetic test signal for demonstration."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [440, 660, 880]  # A4, approx E5, A5
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    # Add some rhythm by modulating amplitude
    amplitude = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow modulation
    y *= amplitude
    # Add noise for broader sound handling
    noise = np.random.normal(0, 0.1, y.shape)
    y += noise
    return y, sr

def process_audio(y, sr):
    """Process audio to compute CQT and improved rhythm features."""
    hop_length = 512
    # Compute CQT with improved parameters for invariance (e.g., more bins per octave)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))
    print("CQT shape:", cqt.shape)
    
    # Onset strength for rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    
    # Tempo estimation (fixed import path)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print("Estimated tempo:", tempo)
    
    # Improved beat tracking for rhythm lattice
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    print("Beat frames:", beats[:10])  # Print first 10 for brevity
    
    # Compute beat-synchronous chroma for coherence and invariance
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.mean)
    print("Synced chroma shape:", chroma_sync.shape)
    
    # Additional rhythm lattice: compute MFCCs and delta features for broader sound handling
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    print("MFCC shape:", mfcc.shape)
    print("Delta MFCC shape:", delta_mfcc.shape)
    
    # For CQT invariance, compute log-amplitude CQT
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    print("Log CQT shape:", log_cqt.shape)

if __name__ == "__main__":
    wav_files = []  # From instructions: (none)
    if not wav_files:
        print("No WAV files available, using synthetic test signal.")
        y, sr = generate_synthetic_signal()
    else:
        # Placeholder for loading actual files (not used here)
        y, sr = librosa.load(wav_files[0], sr=22050)
    
    process_audio(y, sr)