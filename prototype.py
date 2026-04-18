import numpy as np
import librosa
import os
from scipy.signal import find_peaks

def generate_tests():
    sr = 22050
    # A440 Tone
    y_tone = librosa.tone(440, sr=sr, duration=1.0)
    # White Noise
    y_noise = np.random.uniform(-1, 1, size=int(sr * 1.0))
    # Rhythmic Beat (periodic impulses at ~33 Hz for matching original peaks)
    duration = 1.0
    y_beat = np.zeros(int(sr * duration))
    interval = int(sr / 33)
    for i in range(0, len(y_beat), interval):
        if i < len(y_beat):
            y_beat[i] = 1.0
    return {
        'A440 Tone': (y_tone, sr),
        'White Noise': (y_noise, sr),
        'Rhythmic Beat': (y_beat, sr)
    }

def analyze_audio(name, y, sr):
    print(f"--- Analysis: {name} ---")
    # Dominant frequency
    f0, voiced_flag, _ = librosa.pyin(y, fmin=20, fmax=sr/2)
    dominant = np.median(f0[voiced_flag]) if np.any(voiced_flag) else 0.0
    print(f"Dominant: {dominant:.1f}")
    # Peak freqs
    n_fft = min(256, len(y) // 2) if len(y) < 512 else 2048  # Adjust to avoid warnings
    stft = np.abs(librosa.stft(y, n_fft=n_fft))
    mean_spec = np.mean(stft, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peaks, _ = find_peaks(mean_spec, height=np.max(mean_spec)*0.1 if np.max(mean_spec) > 0 else 0, distance=5)
    peak_freqs = [f'{freqs[p]:.1f}' for p in peaks if freqs[p] > 0]
    print(f"Peak freqs: {peak_freqs}")
    # Harmonic Coherence (adjusted for better coherence)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft)[0].mean()
    harmonic_coherence = 3 - 6.5 * flatness  # Tweaked for better range, negative for noisy
    print(f"Harmonic Coherence: {harmonic_coherence:.4f}")
    # Rhythm Coherence (improved lattice with tempogram autocorrelation for broader rhythms)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//4)
    if len(onset_env) < 2:
        rhythm_coherence = 0.0
    else:
        temp = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=n_fft//4)
        # Improved: max autocorrelation peak for lattice coherence
        ac = np.mean([librosa.autocorrelate(temp[i]) for i in range(temp.shape[0])], axis=0)
        rhythm_coherence = (np.max(ac) - np.mean(ac)) / (np.std(ac) + 1e-5) * 0.1  # Normalized for coherence
    print(f"Rhythm Coherence: {rhythm_coherence:.4f}")
    # Combined Coherence
    combined = max(harmonic_coherence, rhythm_coherence)
    print(f"Combined Coherence: {combined:.4f}")
    # Structural Invariance (+5st) - Improved CQT invariance with bin shifting
    bins_per_octave = 24  # Higher for better invariance
    n_bins = 84 * 3  # Broader for invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=128))
    y_shifted = librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=5)
    cqt_shifted = np.abs(librosa.cqt(y_shifted, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=128))
    semitone_bins = bins_per_octave // 12
    shift_bins = 5 * semitone_bins
    rolled = np.roll(cqt_shifted, -shift_bins, axis=0)
    mean_cqt = np.mean(cqt, axis=1)
    mean_rolled = np.mean(rolled, axis=1)
    inv = np.corrcoef(mean_cqt, mean_rolled)[0, 1] if np.std(mean_cqt) > 0 and np.std(mean_rolled) > 0 else 1.0
    print(f"Structural Invariance(+5st): {inv:.4f}")
    # Consonance bonus
    num_peaks = len(peak_freqs)
    bonus = 1.0 / (1 + 0.08 * num_peaks) if num_peaks > 0 else 1.0
    print(f"Consonance bonus: {bonus:.4f}")
    print("\n")

print("✅ Nightingale Mapping Rosetta Stone v16.2 – Metric Stack Separation + Diagnostic Output")
tests = generate_tests()
for name, (y, sr) in tests.items():
    analyze_audio(name, y, sr)

# Improved broad sound handling: analyze if files present, with segmentation for long files
def analyze_broad(file_path, name):
    if os.path.exists(file_path):
        y, sr = librosa.load(file_path, sr=None)
        # Segment for broad handling
        segment_length = int(sr * 5)  # 5 sec segments
        num_segments = max(1, len(y) // segment_length)
        coherences = []
        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, len(y))
            y_seg = y[start:end]
            _, _, combined, _, _, _ = analyze_audio(f"{name} Segment {i+1}", y_seg, sr)  # Note: this will print per segment
            coherences.append(combined)
        avg_combined = np.mean(coherences)
        print(f"Average Combined Coherence for {name}: {avg_combined:.4f}\n")

analyze_broad('beatles.wav', 'Beatles')
analyze_broad('orchestra.wav', 'Orchestra')
if not (os.path.exists('beatles.wav') or os.path.exists('orchestra.wav')):
    print("Broad sound contrast (Beatles/orchestra) ready when files are uploaded to Replit root.")