import numpy as np
import librosa
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("✅ Nightingale Mapping Rosetta Stone v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling")

# Parameters
sr = 22050
duration = 0.1  # Increased duration to handle FFT/CQT better
n_fft = 256
hop_length = 64
n_bins = 84  # For CQT
min_freq = 32.7  # C2

def generate_a440():
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * 440 * t)

def generate_white_noise():
    return np.random.randn(int(sr * duration))

def generate_rhythmic_beat():
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beat = np.zeros_like(t)
    beat_positions = np.arange(0, len(t), int(sr * 0.02))  # Beats every 20ms
    beat[beat_positions] = 1.0
    return beat + 0.1 * np.random.randn(len(t))  # Add some noise

def classify_sound_type(y, sr):
    # Harmonic-percussive separation
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.mean(y_harm**2)
    perc_energy = np.mean(y_perc**2)
    total_energy = harm_energy + perc_energy
    if total_energy == 0:
        return "noise-like"
    harm_ratio = harm_energy / total_energy
    if harm_ratio > 0.5:
        return "pitched"
    else:
        return "noise-like"

def compute_dominant_freq(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=min_freq, n_bins=n_bins)
    cqt_mag = np.abs(cqt)
    avg_spectrum = np.mean(cqt_mag, axis=1)
    bin_idx = np.argmax(avg_spectrum)
    freqs = librosa.cqt_frequencies(n_bins, fmin=min_freq)
    return freqs[bin_idx]

def compute_harmonic_coherence(y, sr, dominant):
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=min_freq, n_bins=n_bins)
    cqt_mag = np.abs(cqt)
    freqs = librosa.cqt_frequencies(n_bins, fmin=min_freq)
    log_freqs = np.log2(freqs)
    dominant_bin = np.argmin(np.abs(log_freqs - np.log2(dominant)))
    
    coherence = 0
    for harm in range(1, 6):  # Check first 5 harmonics
        harm_bin = np.argmin(np.abs(log_freqs - (log_freqs[dominant_bin] + np.log2(harm))))
        coherence += cqt_mag[harm_bin].mean() / (harm * cqt_mag[dominant_bin].mean() + 1e-6)
    return min(1.0, coherence / 3.0)  # Normalized

def compute_rhythm_coherence(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if len(onset_env) < 2:
        return 0.0
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    ac = librosa.autocorrelate(onset_env)
    ac = ac / (np.max(ac) + 1e-6)
    rhythm_coh = np.max(ac[1:]) if len(ac) > 1 else 0.0  # Max autocorrelation peak
    return rhythm_coh

def compute_final_coherence(harm_coh, rhythm_coh, sound_type):
    if sound_type == "pitched":
        return max(harm_coh, rhythm_coh)
    else:
        return rhythm_coh

def compute_structural_invariance(y, sr, semitones=5):
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    cqt_orig = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=min_freq, n_bins=n_bins))
    cqt_shift = np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=hop_length, fmin=min_freq, n_bins=n_bins))
    # Roll the shifted CQT to align fundamentals (approx)
    shift_bins = int(semitones)
    cqt_shift_rolled = np.roll(cqt_shift, -shift_bins, axis=0)
    diff = np.mean(np.abs(cqt_orig - cqt_shift_rolled)) / (np.mean(cqt_orig) + 1e-6)
    return 1.0 - min(1.0, diff)

def compute_consonance_bonus(y, sr):
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    return 1.0 - np.mean(flatness)  # Inverse flatness as consonance

def determine_applicable_metrics(sound_type):
    if sound_type == "pitched":
        return "harmonic+rhythm"
    else:
        return "rhythm-only"

def analyze_sound(name, generate_func):
    y = generate_func()
    sound_type = classify_sound_type(y, sr)
    dominant = compute_dominant_freq(y, sr)
    harm_coh = compute_harmonic_coherence(y, sr, dominant)
    rhythm_coh = compute_rhythm_coherence(y, sr)
    final_coh = compute_final_coherence(harm_coh, rhythm_coh, sound_type)
    invariance = compute_structural_invariance(y, sr)
    consonance = compute_consonance_bonus(y, sr)
    metrics = determine_applicable_metrics(sound_type)
    
    print(f"\n--- Analysis: {name} ---")
    print(f"Sound type: {sound_type}")
    print(f"Dominant: {dominant:.1f}")
    print(f"Harmonic Coherence: {harm_coh:.4f}")
    print(f"Rhythm Coherence: {rhythm_coh:.4f}")
    print(f"Final Coherence: {final_coh:.4f}")
    print(f"Structural Invariance(+5st): {invariance:.4f}")
    print(f"Consonance bonus: {consonance:.4f}")
    print(f"Applicable metrics: {metrics}")

# Run analyses
analyze_sound("A440 Tone", generate_a440)
analyze_sound("White Noise", generate_white_noise)
analyze_sound("Rhythmic Beat", generate_rhythmic_beat)

# Broad sound handling (example placeholder for uploaded files)
print("\nBroad sound contrast (Beatles/orchestra) ready when files are uploaded to Replit root.")
try:
    y_bea, sr_bea = librosa.load("beatles.wav", sr=sr)
    print("Beatles file loaded successfully.")
except FileNotFoundError:
    print("Beatles file not found.")
try:
    y_orc, sr_orc = librosa.load("orchestra.wav", sr=sr)
    print("Orchestra file loaded successfully.")
except FileNotFoundError:
    print("Orchestra file not found.")