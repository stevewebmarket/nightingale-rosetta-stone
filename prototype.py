# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.2 – Metric Stack Separation: Structural Invariance + Diagnostic Output
# =============================================================================

import numpy as np
import librosa
from scipy.signal import find_peaks
from copy import deepcopy

VERSION = "v16.2"

def generate_a440(duration=1, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    return np.sin(2 * np.pi * 440 * t)

def generate_white_noise(duration=1, sr=22050):
    return np.random.uniform(-1, 1, int(sr * duration))

def generate_rhythmic_beat(duration=1, sr=22050):
    beat_freq = 32.7
    period = 1 / beat_freq
    num_samples = int(sr * duration)
    signal = np.zeros(num_samples)
    pos = 0
    while pos < num_samples:
        if pos < num_samples:
            signal[pos] = 1.0
        pos += int(sr * period)
    return signal

def get_cqt_freqs(n_bins=96, bins_per_octave=12, fmin=librosa.note_to_hz('C0')):
    return fmin * 2 ** (np.arange(n_bins) / bins_per_octave)

# Structural invariance (CQT-based, shift-tolerant)
def compute_structural_invariance(cqt_mag, shift_steps=5):
    rolled = np.roll(cqt_mag, -shift_steps, axis=0)
    a = np.mean(cqt_mag, axis=1)
    b = np.mean(rolled, axis=1)
    corr = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return max(corr, 0.0)  # structural, not anti-correlation

def compute_coherence(mean_spec):
    max_val = np.max(mean_spec)
    avg_val = np.mean(mean_spec)
    return (max_val - avg_val) / (max_val + 1e-6)

def compute_consonance_bonus(peak_freqs):
    if not peak_freqs:
        return 1.0000
    num_peaks = len(peak_freqs)
    return 1 / (1 + 0.08 * num_peaks)

def analyze_rhythm(signal, sr):
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr)
    onsets = librosa.onset.onset_detect(y=signal, sr=sr)
    if len(onsets) > 1:
        diffs = np.diff(onsets)
        rhythm_coherence = 1 - np.std(diffs) / np.mean(diffs) if np.mean(diffs) > 0 else 0.0
    else:
        rhythm_coherence = 0.0
    return max(rhythm_coherence, 0.0), tempo

def analyze(signal, sr=22050, label="segment"):
    fmin = librosa.note_to_hz('C0')
    cqt = librosa.cqt(signal, sr=sr, fmin=fmin, n_bins=96, bins_per_octave=12, hop_length=256)
    cqt_mag = np.abs(cqt)
    log_cqt = librosa.amplitude_to_db(cqt_mag)
    mean_spec = np.mean(log_cqt, axis=1)
   
    freqs = get_cqt_freqs(n_bins=96, bins_per_octave=12, fmin=fmin)
   
    dominant_idx = np.argmax(mean_spec)
    window = slice(max(0, dominant_idx-3), dominant_idx+4)
    dominant = np.average(freqs[window], weights=mean_spec[window])
   
    max_val = np.max(mean_spec)
    peaks, _ = find_peaks(mean_spec, height=max_val - 30, prominence=10)
    peak_freqs = [f'{f:.1f}' for f in freqs[peaks]]
   
    spectral_coherence = compute_coherence(mean_spec)
    rhythm_coherence, tempo = analyze_rhythm(signal, sr)
   
    structural_invariance = compute_structural_invariance(cqt_mag)
    consonance = compute_consonance_bonus(peak_freqs)
   
    # Separate harmonic and rhythmic coherence for diagnostics
    harmonic_coherence = spectral_coherence
    combined_coherence = max(spectral_coherence, rhythm_coherence)
   
    print(f"\n--- Analysis: {label} ---")
    print(f"Dominant: {dominant:.1f}")
    print(f"Harmonic Coherence: {harmonic_coherence:.4f}")
    print(f"Rhythm Coherence: {rhythm_coherence:.4f}")
    print(f"Combined Coherence: {combined_coherence:.4f}")
    print(f"Structural Invariance(+5st): {structural_invariance:.4f}")
    print(f"Peak freqs: {peak_freqs}")
    print(f"Consonance bonus: {consonance:.4f}\n")
   
    return {
        'dominant': dominant,
        'harmonic_coherence': harmonic_coherence,
        'rhythm_coherence': rhythm_coherence,
        'combined_coherence': combined_coherence,
        'invariance': structural_invariance,
        'consonance': consonance,
        'tempo': tempo
    }

if __name__ == "__main__":
    sr = 22050
    duration = 1.0
    print(f"✅ Nightingale Mapping Rosetta Stone {VERSION} – Metric Stack Separation + Diagnostic Output")
   
    for name, gen in [
        ("A440 Tone", generate_a440),
        ("White Noise", generate_white_noise),
        ("Rhythmic Beat", generate_rhythmic_beat)
    ]:
        signal = gen(duration=duration, sr=sr)
        signal = librosa.util.normalize(signal)
        analyze(signal, sr, label=name)
   
    print("Broad sound contrast (Beatles/orchestra) ready when files are uploaded to Replit root.")
