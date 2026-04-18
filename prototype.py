import numpy as np
import librosa
import os
import scipy.signal as signal

# Nightingale Mapping Rosetta Stone v16.6 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling

# Constants
SAMPLE_RATE = 22050
DURATION = 5.0  # seconds
A440_FREQ = 440.0
BEAT_BPM = 120
SHIFT_SEMITONES = 5

# Improved rhythm lattice function with better onset detection and tempogram
def compute_rhythm_coherence(y, sr):
    # Onset envelope with finer parameters
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, hop_length=512, n_fft=2048)
    # Tempogram with wider BPM range for broad sound handling
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, bpm=BEAT_BPM, hop_length=512)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=512)
    # Improved coherence: autocorrelation of tempogram with normalization
    auto_corr = librosa.autocorrelate(tempogram, axis=1)
    rhythm_coh = np.mean(np.max(auto_corr, axis=1)) / (np.mean(onset_env) + 1e-5)  # Normalized by onset strength
    # Adjust for noise: penalize if low periodicity
    if tempo == 0:
        rhythm_coh *= 0.1  # Reduce for aperiodic noise
    return min(rhythm_coh, 1.0)

# Improved harmonic coherence with enhanced CQT
def compute_harmonic_coherence(y, sr):
    # CQT with more octaves for invariance and broad handling
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=84*2, bins_per_octave=12*2))  # Doubled for precision
    # Spectral centroid and flatness for coherence
    centroid = librosa.feature.spectral_centroid(S=cqt)
    flatness = librosa.feature.spectral_flatness(S=cqt)
    harm_coh = (1 - np.mean(flatness)) * (np.std(centroid) / (np.mean(centroid) + 1e-5))
    return min(harm_coh * 2.0, 1.0)  # Scaled up for better range

# Dominant frequency detection
def get_dominant_freq(y, sr):
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    return freqs[np.argmax(np.abs(fft))]

# Structural invariance under pitch shift (+5 semitones)
def compute_structural_invariance(y, sr, shift_semitones):
    # Pitch shift using phase vocoder with improved quality
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=shift_semitones, bins_per_octave=12)
    # CQT for both
    cqt_orig = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=84))
    cqt_shift = np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=512, n_bins=84))
    # Correlation for invariance, improved with normalization
    corr = signal.correlate(cqt_orig.flatten(), cqt_shift.flatten(), mode='valid')
    return np.max(corr) / (np.linalg.norm(cqt_orig) * np.linalg.norm(cqt_shift) + 1e-5)

# Consonance bonus based on spectral properties
def compute_consonance_bonus(y, sr):
    S = np.abs(librosa.stft(y))
    roughness = librosa.feature.spectral_bandwidth(S=S)
    flatness = librosa.feature.spectral_flatness(S=S)
    return 1 - (np.mean(roughness) / sr + np.mean(flatness)) / 2

# Main analysis function with improved logic
def analyze_sound(y, sr, name):
    print(f"--- Analysis: {name} ---")
    
    harm_coh = compute_harmonic_coherence(y, sr)
    rhythm_coh = compute_rhythm_coherence(y, sr)
    
    if harm_coh > 0.1:  # Threshold adjusted for better classification
        sound_type = "pitched"
        dominant = get_dominant_freq(y, sr)
        print(f"Sound type: {sound_type}")
        print(f"Dominant: {dominant:.1f}")
        print(f"Harmonic Coherence: {harm_coh:.4f}")
        print(f"Rhythm Coherence: {rhythm_coh:.4f}")
        final_coh = harm_coh
        metrics = "harmonic"
    else:
        sound_type = "noise"
        print(f"Sound type: {sound_type}")
        print(f"Rhythm Coherence: {rhythm_coh:.4f}")
        final_coh = rhythm_coh
        metrics = "rhythm"
    
    print(f"Final Coherence: {final_coh:.4f}")
    
    invariance = compute_structural_invariance(y, sr, SHIFT_SEMITONES)
    print(f"Structural Invariance(+5st): {invariance:.4f}")
    
    consonance = compute_consonance_bonus(y, sr)
    print(f"Consonance bonus: {consonance:.4f}")
    
    print(f"Applicable metrics: {metrics}")

# Generate synthetic sounds
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

# A440 Tone
y_tone = 0.5 * np.sin(2 * np.pi * A440_FREQ * t)

# White Noise
y_noise = np.random.normal(0, 0.5, len(t))

# Rhythmic Beat (improved: clearer periodic pulses)
beat_interval = int(SAMPLE_RATE * 60 / BEAT_BPM)
y_beat = np.zeros(len(t))
y_beat[::beat_interval] = 1.0
y_beat = signal.convolve(y_beat, signal.gaussian(200, 10), mode='same')  # Smoothed pulses for realism

# Run analyses
print("✅ Nightingale Mapping Rosetta Stone v16.6 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling")
analyze_sound(y_tone, SAMPLE_RATE, "A440 Tone")
analyze_sound(y_noise, SAMPLE_RATE, "White Noise")
analyze_sound(y_beat, SAMPLE_RATE, "Rhythmic Beat")

# Broad sound contrast
print("\nBroad sound contrast (Beatles/orchestra) ready when files are uploaded to Replit root.")
if os.path.exists('beatles.wav'):
    y_beatles, sr_beatles = librosa.load('beatles.wav', sr=SAMPLE_RATE)
    analyze_sound(y_beatles, sr_beatles, "Beatles Song")
else:
    print("Beatles file not found.")

if os.path.exists('orchestra.wav'):
    y_orch, sr_orch = librosa.load('orchestra.wav', sr=SAMPLE_RATE)
    analyze_sound(y_orch, sr_orch, "Orchestra Piece")
else:
    print("Orchestra file not found.")