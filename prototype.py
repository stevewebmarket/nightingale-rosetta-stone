import numpy as np
import librosa
from scipy.signal import find_peaks

def pad_signal(y, min_len=2048):
    if len(y) < min_len:
        pad = min_len - len(y)
        y = np.pad(y, (0, pad), 'constant')
    return y

def analyze(y, sr=22050):
    y = pad_signal(y)
    cqt = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    cqt_mag = np.abs(cqt)
    mean_spec = np.mean(cqt_mag, axis=1)
    freqs = librosa.cqt_frequencies(168, fmin=librosa.note_to_hz('C1'), bins_per_octave=24)

    # Dominant
    dominant_bin = np.argmax(mean_spec)
    dominant = freqs[dominant_bin]

    # Improved coherence: log of peak to mean ratio
    mean_val = np.mean(mean_spec) + 1e-10
    coherence = np.log(np.max(mean_spec) / mean_val + 1e-10)

    # Invariance with proper pitch shift and roll
    st = 5
    y_shift = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=st, bins_per_octave=24)
    cqt_shift = librosa.cqt(y_shift, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=168, bins_per_octave=24)
    mean_spec_shift = np.mean(np.abs(cqt_shift), axis=1)
    shift_amount = st * (24 // 12)
    shifted_mean_spec = np.roll(mean_spec, shift_amount)
    invariance = np.corrcoef(shifted_mean_spec, mean_spec_shift)[0, 1]

    # Peak freqs
    peaks, _ = find_peaks(mean_spec, height=np.max(mean_spec) * 0.1)
    peak_freqs = freqs[peaks]
    peak_freq_str = [f'{f:.1f}' for f in peak_freqs]

    # Consonance bonus
    num_peaks = len(peak_freq_str)
    consonance = 1.0 / (1.0 + num_peaks * 0.1) if (1.0 + num_peaks * 0.1) > 0 else 1.0

    return dominant, coherence, invariance, peak_freq_str, consonance

def print_analysis(name, dominant, coherence, invariance, peak_freq_str, consonance):
    print(f"--- Analysis: {name} ---")
    print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {invariance:.4f}")
    print(f"Peak freqs: {peak_freq_str}")
    print(f"Consonance bonus: {consonance:.4f}\n")

if __name__ == "__main__":
    print("✅ Nightingale Mapping Rosetta Stone v18.0 – Improved Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling")
    sr = 22050
    length = 173
    t = np.arange(length) / sr

    # A440 Tone
    y_a440 = np.sin(2 * np.pi * 440 * t)
    dom, coh, inv, peaks, cons = analyze(y_a440, sr)
    print_analysis("A440 Tone", dom, coh, inv, peaks, cons)

    # White Noise
    np.random.seed(42)  # For reproducibility
    y_white = np.random.randn(length)
    dom, coh, inv, peaks, cons = analyze(y_white, sr)
    print_analysis("White Noise", dom, coh, inv, peaks, cons)

    # Rhythmic Beat (improved to better capture lattice with harmonic components)
    rhythm_freqs = [32.7 * i for i in range(1, 6)]
    y_rhythm = np.sum([np.sin(2 * np.pi * f * t) for f in rhythm_freqs], axis=0)
    dom, coh, inv, peaks, cons = analyze(y_rhythm, sr)
    print_analysis("Rhythmic Beat", dom, coh, inv, peaks, cons)