import numpy as np
from scipy.signal import find_peaks
import librosa

# Nightingale Mapping Rosetta Stone - Prototype
# Version 16.5: Improved rhythm lattice with better peak grouping, fixed coherence calculation to ensure non-negative,
# enhanced CQT invariance via better pitch shift similarity, broader sound handling with adaptive thresholding.

VERSION = "16.5"

def generate_a440_tone(sr=22050, duration=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t)

def generate_white_noise(duration=1.0, sr=22050):
    return np.random.normal(0, 1, int(sr * duration))

def generate_rhythmic_beat(sr=22050, duration=1.0, bpm=120):
    beat_interval = sr / (bpm / 60)
    signal = np.zeros(int(sr * duration))
    for i in range(int(duration * (bpm / 60))):
        start = int(i * beat_interval)
        signal[start:start+100] = 1.0  # Simple pulse
    return signal

def analyze_signal(signal, sr=22050):
    # Compute CQT
    cqt = np.abs(librosa.cqt(signal, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12))
    cqt_mean = np.mean(cqt, axis=1)
    
    # Find peaks with adaptive thresholding
    height_threshold = np.max(cqt_mean) * 0.1
    peaks, _ = find_peaks(cqt_mean, height=height_threshold, distance=3)
    peak_freqs = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'), bins_per_octave=12)[peaks]
    peak_freqs = [f'{f:.1f}' for f in peak_freqs if f < 500]  # Limit for display
    
    # Dominant frequency
    dominant_idx = np.argmax(cqt_mean)
    dominant = librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz('C1'), bins_per_octave=12)[dominant_idx]
    
    # Improved coherence: normalized peak energy ratio, ensure non-negative
    total_energy = np.sum(cqt_mean)
    peak_energy = np.sum(cqt_mean[peaks])
    coherence = np.clip(peak_energy / total_energy if total_energy > 0 else 0, 0, 1)
    
    # CQT Invariance: shift by +5 semitones and compute similarity
    shift_factor = 2 ** (5/12)
    signal_shifted = librosa.effects.pitch_shift(signal, sr=sr, n_steps=5)
    cqt_shifted = np.abs(librosa.cqt(signal_shifted, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12))
    cqt_shifted_mean = np.mean(cqt_shifted, axis=1)
    # Roll the original to match shift
    bins_per_semitone = 1
    rolled_cqt = np.roll(cqt_mean, -5 * bins_per_semitone)
    invariance = np.corrcoef(rolled_cqt, cqt_shifted_mean)[0, 1]
    
    # Consonance bonus: simple ratio based on number of peaks (improved for lattice)
    consonance = 1 / (1 + len(peaks) / 10.0)  # Fewer peaks = higher consonance
    
    # Improved rhythm lattice for rhythmic signals: group peaks into lattice
    if len(peak_freqs) > 10:  # Detect rhythmic
        diffs = np.diff([float(f) for f in peak_freqs])
        lattice_period = np.median(diffs)
        grouped_peaks = [peak_freqs[0]]
        for p in peak_freqs[1:]:
            if float(p) - float(grouped_peaks[-1]) > lattice_period * 0.5:
                grouped_peaks.append(p)
        peak_freqs = grouped_peaks
        coherence = min(1.0, coherence * 1.5)  # Boost coherence for lattice
    
    return dominant, coherence, invariance, peak_freqs, consonance

def main():
    sr = 22050
    print(f"✅ Nightingale Mapping Rosetta Stone v{VERSION} – Enhanced Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling")
    
    # Analysis: A440 Tone
    signal = generate_a440_tone(sr=sr)
    dominant, coherence, invariance, peak_freqs, consonance = analyze_signal(signal, sr)
    print("--- Analysis: A440 Tone ---")
    print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {invariance:.4f}")
    print(f"Peak freqs: {peak_freqs}")
    print(f"Consonance bonus: {consonance:.4f}\n")
    
    # Analysis: White Noise
    signal = generate_white_noise(sr=sr)
    dominant, coherence, invariance, peak_freqs, consonance = analyze_signal(signal, sr)
    print("--- Analysis: White Noise ---")
    print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {invariance:.4f}")
    print(f"Peak freqs: {peak_freqs}")
    print(f"Consonance bonus: {consonance:.4f}\n")
    
    # Analysis: Rhythmic Beat
    signal = generate_rhythmic_beat(sr=sr)
    dominant, coherence, invariance, peak_freqs, consonance = analyze_signal(signal, sr)
    print("--- Analysis: Rhythmic Beat ---")
    print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {invariance:.4f}")
    print(f"Peak freqs: {peak_freqs}")
    print(f"Consonance bonus: {consonance:.4f}\n")

if __name__ == "__main__":
    main()