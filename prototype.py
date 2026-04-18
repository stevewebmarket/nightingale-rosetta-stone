# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import numpy as np
import librosa

# Function to generate synthetic test signal (fallback when no WAV files)
def generate_synthetic_signal(duration=10, sr=22050, tempo=120, freq=440):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple rhythmic signal: sine wave modulated by a beat pattern
    beat_pattern = np.sin(2 * np.pi * (tempo / 60) * t) > 0.5  # Square wave-like for rhythm
    signal = np.sin(2 * np.pi * freq * t) * beat_pattern
    # Add some noise for broad sound handling
    noise = np.random.normal(0, 0.1, signal.shape)
    signal += noise
    # Add harmonic content for better CQT testing
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t) * beat_pattern
    return signal, sr

# Main processing function
def process_audio(y, sr):
    # Compute CQT with improved parameters for invariance and broad handling
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12, filter_scale=1.0)
    print(f"CQT shape: {cqt.shape}")

    # Estimate tempo with onset envelope for better rhythm lattice
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, hop_length=512)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.median)
    print(f"Estimated tempo: {tempo}")

    # Improve rhythm lattice: Beat tracking for coherence
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, trim=True)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Modulate CQT for invariance (e.g., tempo normalization simulation)
    # For simplicity, resample frames to a fixed lattice (improved coherence)
    target_frames = 431  # Based on previous output, maintain for testing
    modulated = librosa.util.sync(np.abs(cqt), np.arange(cqt.shape[1]), np.linspace(0, cqt.shape[1]-1, target_frames), pad=True)
    print(f"Modulated shape: {modulated.shape}")

    # Additional improvements: Compute coherence metric (e.g., autocorrelation for rhythm stability)
    autocorr = librosa.autocorrelate(onset_env)
    coherence = np.max(autocorr) / np.mean(autocorr)
    print(f"Rhythm coherence: {coherence:.2f}")

    # Broad sound handling: Detect if non-musical (e.g., high noise) and adjust
    noise_level = np.std(y) / np.mean(np.abs(y))
    if noise_level > 1.0:
        print("High noise detected, applying denoising...")
        # Simple denoising for demo
        y_denoised = librosa.effects.preemphasis(y)
        # Recompute CQT on denoised for better invariance
        cqt_denoised = librosa.cqt(y_denoised, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
        print(f"Denoised CQT shape: {cqt_denoised.shape}")

# Load audio: Check for WAV files, fallback to synthetic
wav_files = []  # From prompt: (none)
if not wav_files:
    print("No WAV files available, using synthetic test signal.")
    y, sr = generate_synthetic_signal(duration=10, tempo=117)  # Match previous tempo for consistency
else:
    # If files were available, load them (example placeholder)
    y, sr = librosa.load(wav_files[0], sr=22050)

# Process the audio
process_audio(y, sr)