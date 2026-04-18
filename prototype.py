# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed tempo estimation, enhanced rhythm lattice with tempogram, improved CQT invariance via normalization, broader sound handling with noise robustness
# =============================================================================

import librosa
import numpy as np
import os

def generate_synthetic_signal(sr=22050, duration=10.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    frequencies = [440, 880, 220]  # A4, A5, A3 for variety
    y = np.sum([np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)
    # Add some noise for broad sound handling
    noise = np.random.normal(0, 0.05, len(y))
    y += noise
    return y

def process_audio(y, sr):
    hop_length = 512
    # Improved CQT: use log amplitude for better invariance to amplitude changes
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12))
    cqt = librosa.amplitude_to_db(cqt)  # Log scale for invariance
    print("CQT shape:", cqt.shape)
    
    # Onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Fixed tempo estimation
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print("Estimated tempo:", tempo)
    
    # Improved rhythm lattice: compute tempogram for better rhythm representation
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    # Enhance coherence by averaging over time for a lattice summary
    rhythm_lattice = np.mean(tempogram, axis=1)
    print("Rhythm lattice shape:", rhythm_lattice.shape)
    
    # Broad sound handling: apply median filter for noise robustness
    rhythm_lattice = scipy.signal.medfilt(rhythm_lattice, kernel_size=3)
    
    # Further processing (placeholder for mapping)
    print("Processing complete.")

if __name__ == "__main__":
    wav_files = []  # From provided list (none)
    if not wav_files:
        print("No WAV files available, using synthetic test signal.")
        sr = 22050
        y = generate_synthetic_signal(sr=sr)
        process_audio(y, sr)
    else:
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                y, sr = librosa.load(wav_file, sr=22050)
                process_audio(y, sr)
            else:
                print(f"File not found: {wav_file}")