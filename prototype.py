import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine
from scipy import signal

print("✅ Nightingale Mapping Rosetta Stone v16.1 – Full Autonomous Agent Takeover Ready")
print("\n✅ v16.1 loaded – Full Agent Takeover Ready.")
print("File is always 'prototype.py'.")
print("Type 'iterate' for v16.2 or use the Replit agent below.")
print("\nDemo analysis:\n")

# Improved generate_tone with optional harmonics and noise for broad sound handling
def generate_tone(freq=440, duration=1, sr=22050, harmonics=True, noise_level=0.01):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    if harmonics:
        y = signal.sawtooth(2 * np.pi * freq * t)  # Sawtooth for harmonics
    else:
        y = np.sin(2 * np.pi * freq * t)
    # Add noise for broad sound handling testing
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

# Improved analyze_audio with CQT for invariance, rhythm lattice, coherence
def analyze_audio(y, sr, shift_steps=5):
    # CQT parameters for better invariance (12 bins per octave for semitone resolution)
    bins_per_octave = 12
    n_bins = 84  # Covering typical range
    fmin = librosa.note_to_hz('C1')
    cqt = librosa.cqt(y, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=fmin)
    cqt_mag = np.abs(cqt)
    spec = np.mean(cqt_mag, axis=1)  # Time-averaged spectrum

    # Pitch-shifted version
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_steps)
    cqt_shifted = librosa.cqt(y_shifted, sr=sr, bins_per_octave=bins_per_octave, n_bins=n_bins, fmin=fmin)
    spec_shifted = np.mean(np.abs(cqt_shifted), axis=1)

    # Improved CQT invariance: roll back by shift_steps bins
    spec_shifted_rolled = np.roll(spec_shifted, -shift_steps)
    # Cosine similarity for invariance measure
    invariance = 1 - cosine(spec + 1e-10, spec_shifted_rolled + 1e-10)  # Avoid div by zero

    # For dominant freq and peaks, use STFT for precision
    stft = librosa.stft(y)
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.mean(np.abs(stft), axis=1)
    peaks, props = find_peaks(mag, height=np.max(mag) * 0.05, prominence=0.1)
    peak_freqs = freqs[peaks]
    # Improve precision with parabolic interpolation approximation
    refined_peaks = []
    for i in peaks:
        if i > 0 and i < len(mag) - 1:
            alpha = mag[i-1]
            beta = mag[i]
            gamma = mag[i+1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-10)
            refined_freq = freqs[i] + p * (freqs[1] - freqs[0])
            refined_peaks.append(refined_freq)
    peak_freqs = np.sort(refined_peaks)[:12]  # Top 12 for display
    dominant = peak_freqs[0] if len(peak_freqs) > 0 else 0

    # Improved coherence: harmonic coherence (energy in harmonic positions)
    if dominant > 0:
        harmonic_indices = np.round((dominant * np.arange(1, 11)) / (freqs[1] - freqs[0])).astype(int)
        harmonic_indices = harmonic_indices[harmonic_indices < len(mag)]
        peak_energy = np.sum(mag[harmonic_indices] ** 2)
        total_energy = np.sum(mag ** 2) + 1e-10
        coherence = peak_energy / total_energy
    else:
        coherence = 0

    # Consonance bonus (simple: based on number of harmonic peaks)
    consonance = min(1.0, len(peak_freqs) / 10.0)

    # Improved rhythm lattice: basic onset and beat tracking for rhythmic structure
    onsets = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onsets, sr=sr)
    rhythm_coherence = len(beats) / duration if duration > 0 else 0  # Simple measure

    return dominant, coherence, invariance, peak_freqs, consonance, rhythm_coherence, tempo

# Demo on A440 tone with improvements
sr = 22050
duration = 1.0
y = generate_tone(440, duration, sr, harmonics=True, noise_level=0.005)  # Added noise for broad handling
dominant, coherence, invariance, peak_freqs, consonance, rhythm_coherence, tempo = analyze_audio(y, sr)

print("--- Analysis: A440 Tone ---")
print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.4f} | Invariance(+5st): {invariance:.4f}")
print(f"Peak freqs: {[f'{p:.1f}' for p in peak_freqs]}")
print(f"Consonance bonus: {consonance:.4f}")
# New: Rhythm lattice info
print(f"Rhythm Lattice: Tempo {tempo:.1f} BPM | Coherence: {rhythm_coherence:.4f}")