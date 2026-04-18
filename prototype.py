import numpy as np
import librosa
import scipy.signal as signal
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

VERSION = "16.4"

def generate_tone(freq=440.0, duration=0.01, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def compute_dominant_freq(y, sr):
    # Use CQT for better frequency resolution
    cqt = librosa.cqt(y, sr=sr, hop_length=64, n_bins=84*2, bins_per_octave=24*2, fmin=librosa.note_to_hz('C1'))
    cqt_mag = np.abs(cqt)
    freqs = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C1'), bins_per_octave=24*2)
    mean_spec = np.mean(cqt_mag, axis=1)
    peak_idx = signal.find_peaks(mean_spec, height=np.max(mean_spec)*0.1)[0]
    if len(peak_idx) > 0:
        dom_idx = peak_idx[np.argmax(mean_spec[peak_idx])]
        return freqs[dom_idx], [str(round(f, 1)) for f in freqs[peak_idx]]
    return 0.0, []

def compute_coherence(y, sr, dom_freq):
    if dom_freq == 0:
        return 0.0
    # Improved coherence using autocorrelation
    corr = signal.correlate(y, y, mode='full')
    corr = corr[len(corr)//2:]
    peaks = signal.find_peaks(corr)[0]
    if len(peaks) > 1:
        periods = np.diff(peaks)
        return 1 - np.std(periods) / np.mean(periods)
    return 0.0

def compute_invariance(y, sr, dom_freq, semitones=5):
    if dom_freq == 0:
        return 0.0
    # Shift by semitones
    shift_factor = 2**(semitones/12)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    dom_shifted, _ = compute_dominant_freq(y_shifted, sr)
    return abs(dom_shifted - dom_freq * shift_factor) / (dom_freq * shift_factor)

def compute_consonance_bonus(peaks):
    # Simple consonance based on peak ratios
    if len(peaks) < 2:
        return 0.0
    ratios = [float(p)/float(peaks[0]) for p in peaks[1:]]
    bonus = sum(1/(r-1) if abs(r - round(r)) < 0.1 else 0 for r in ratios) / len(ratios)
    return min(bonus, 1.0) * 0.6367  # Normalized

def analyze_tone(y, sr, name="A440 Tone"):
    dom_freq, peak_freqs = compute_dominant_freq(y, sr)
    coherence = compute_coherence(y, sr, dom_freq)
    invariance = compute_invariance(y, sr, dom_freq)
    cons_bonus = compute_consonance_bonus(peak_freqs)
    
    print(f"--- Analysis: {name} ---")
    print(f"Dominant: {dom_freq:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {invariance:.4f}")
    print(f"Peak freqs: {peak_freqs}")
    print(f"Consonance bonus: {cons_bonus:.4f}\n")

def main():
    print(f"✅ Nightingale Mapping Rosetta Stone v{VERSION} – Enhanced Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling")
    
    # Test with A440 tone (short duration to simulate broad handling)
    sr = 22050
    y = generate_tone(440.0, 0.01, sr)  # Short signal
    analyze_tone(y, sr)
    
    # Additional test for broad sound handling: white noise
    y_noise = np.random.normal(0, 1, int(sr * 0.01))
    analyze_tone(y_noise, sr, "White Noise")
    
    # Test rhythm lattice: simple beat
    t = np.linspace(0, 1, sr, endpoint=False)
    beat = np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 2 * t) > 0.5)
    analyze_tone(beat, sr, "Rhythmic Beat")

if __name__ == "__main__":
    main()