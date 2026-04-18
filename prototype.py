import numpy as np
import librosa
from scipy.signal import find_peaks

VERSION = "v17.0"

def generate_a440(duration=1, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    return np.sin(2 * np.pi * 440 * t)

def generate_white_noise(duration=1, sr=22050):
    return np.random.uniform(-1, 1, int(sr * duration))

def generate_rhythmic_beat(duration=1, sr=22050):
    beat_freq = 32.7  # Hz for periodic impulses
    period = 1 / beat_freq
    num_samples = int(sr * duration)
    signal = np.zeros(num_samples)
    pos = 0
    while pos < num_samples:
        if pos < num_samples:
            signal[pos] = 1.0
        pos += int(sr * period)
    return signal

def get_cqt_freqs(n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')):
    return fmin * 2 ** (np.arange(n_bins) / bins_per_octave)

def compute_coherence(mean_spec):
    max_val = np.max(mean_spec)
    avg_val = np.mean(mean_spec)
    return (max_val - avg_val) / (max_val + 1e-6)  # Normalized coherence

def compute_invariance(cqt_mag, shift_steps=5):
    # Improved CQT invariance: correlation after rolling bins
    rolled = np.roll(cqt_mag, -shift_steps, axis=0)
    a = np.mean(cqt_mag, axis=1)
    b = np.mean(rolled, axis=1)
    corr = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
    return corr - 1  # Deviation from perfect invariance

def compute_consonance_bonus(peak_freqs):
    if not peak_freqs:
        return 1.0000
    num_peaks = len(peak_freqs)
    return 1 / (1 + 0.1 * num_peaks)  # Simple bonus, adjust for better consonance estimation

def analyze_rhythm(signal, sr):
    # New: Rhythm lattice improvement using tempogram
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr)
    onsets = librosa.onset.onset_detect(y=signal, sr=sr)
    if len(onsets) > 1:
        diffs = np.diff(onsets)
        rhythm_coherence = 1 - np.std(diffs) / np.mean(diffs)  # Low variation means coherent rhythm
    else:
        rhythm_coherence = 0.0
    return max(rhythm_coherence, 0.0), tempo

def analyze(signal, sr=22050):
    # CQT with improved parameters for broad sound handling
    fmin = librosa.note_to_hz('C0')  # Lower fmin for broader range
    cqt = librosa.cqt(signal, sr=sr, fmin=fmin, n_bins=96, bins_per_octave=12, hop_length=256)
    cqt_mag = np.abs(cqt)
    log_cqt = librosa.amplitude_to_db(cqt_mag)
    mean_spec = np.mean(log_cqt, axis=1)
    
    freqs = get_cqt_freqs(n_bins=96, bins_per_octave=12, fmin=fmin)
    
    # Dominant frequency (improved: weighted average around peak)
    dominant_idx = np.argmax(mean_spec)
    window = slice(max(0, dominant_idx-2), dominant_idx+3)
    dominant = np.average(freqs[window], weights=mean_spec[window])
    
    # Peaks with improved threshold for coherence
    max_val = np.max(mean_spec)
    peaks, _ = find_peaks(mean_spec, height=max_val - 30, prominence=10)
    peak_freqs = [f'{f:.1f}' for f in freqs[peaks]]
    
    # Coherence with rhythm integration for improvement
    spectral_coherence = compute_coherence(mean_spec)
    rhythm_coherence, tempo = analyze_rhythm(signal, sr)
    combined_coherence = max(spectral_coherence, rhythm_coherence)  # Boost for rhythmic sounds
    
    # Invariance with improved shift handling
    invariance = compute_invariance(cqt_mag)
    
    # Consonance bonus
    consonance = compute_consonance_bonus(peak_freqs)
    
    return {
        'dominant': dominant,
        'coherence': combined_coherence,
        'invariance': invariance,
        'peak_freqs': peak_freqs,
        'consonance': consonance
    }

if __name__ == "__main__":
    sr = 22050
    duration = 1.0
    print(f"✅ Nightingale Mapping Rosetta Stone {VERSION} – Improved Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling")
    
    for name, gen in [
        ("A440 Tone", generate_a440),
        ("White Noise", generate_white_noise),
        ("Rhythmic Beat", generate_rhythmic_beat)
    ]:
        signal = gen(duration=duration, sr=sr)
        signal = librosa.util.normalize(signal)
        results = analyze(signal, sr)
        
        print(f"--- Analysis: {name} ---")
        print(f"Dominant: {results['dominant']:.1f} | Coherence: {results['coherence']:.3f} | Invariance(+5st): {results['invariance']:.4f}")
        print(f"Peak freqs: {results['peak_freqs']}")
        print(f"Consonance bonus: {results['consonance']:.4f}\n")