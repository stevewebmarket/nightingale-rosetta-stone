# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.5 – Refined Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import numpy as np
import librosa
import os
import warnings
warnings.filterwarnings('ignore')

# Constants
SR = 22050  # Sample rate
DURATION = 2.0  # Duration in seconds
N_FFT = 2048
HOP_LENGTH = 512
CQT_BINS = 84  # For CQT, spanning 7 octaves
CQT_FMIN = librosa.note_to_hz('C2')

# Function to generate test sounds
def generate_a440_tone(sr, duration):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)

def generate_white_noise(sr, duration):
    return np.random.uniform(-1, 1, int(sr * duration))

def generate_rhythmic_beat(sr, duration, bpm=120):
    beat_length = int(sr * 60 / bpm)
    num_beats = int(sr * duration) // beat_length
    signal = np.zeros(int(sr * duration))
    for i in range(num_beats):
        start = i * beat_length
        signal[start:start+beat_length//4] = np.sin(2 * np.pi * 55 * np.linspace(0, beat_length//4 / sr, beat_length//4))
    return signal / np.max(np.abs(signal))

# Improved sound type classification
def classify_sound_type(y, sr):
    # Spectral flatness for noise detection
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flatness = librosa.feature.spectral_flatness(S=stft)
    avg_flatness = np.mean(flatness)
    
    # Harmonic-to-noise ratio approximation
    y_harm, y_perc = librosa.effects.hpss(y)
    hnr = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-8)
    
    if avg_flatness > 0.3 or hnr < 0.5:
        return "noise"
    elif hnr > 2.0:
        return "pitched"
    else:
        return "mixed"

# Dominant frequency using CQT for better invariance
def get_dominant_freq(y, sr):
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN, n_bins=CQT_BINS))
    avg_cqt = np.mean(cqt, axis=1)
    bin_idx = np.argmax(avg_cqt)
    freq = librosa.cqt_frequencies(n_bins=CQT_BINS, fmin=CQT_FMIN)[bin_idx]
    return round(freq, 1)

# Improved harmonic coherence using CQT
def harmonic_coherence(y, sr, dom_freq):
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN, n_bins=CQT_BINS))
    freqs = librosa.cqt_frequencies(n_bins=CQT_BINS, fmin=CQT_FMIN)
    harm_indices = []
    for i in range(1, 6):  # Check first 5 harmonics
        harm_freq = dom_freq * i
        idx = np.argmin(np.abs(freqs - harm_freq))
        if idx < len(freqs):
            harm_indices.append(idx)
    if not harm_indices:
        return 0.0
    harm_amps = np.mean(cqt[harm_indices, :], axis=1)
    coherence = np.mean(harm_amps / (np.max(harm_amps) + 1e-8))
    return round(coherence, 4)

# Refined rhythm lattice coherence using tempogram
def rhythm_coherence(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    auto_corr = librosa.autocorrelate(onset_env)
    rhythm_lattice = np.mean(np.abs(tempogram), axis=0)
    coherence = np.mean(auto_corr[:len(auto_corr)//2]) / (np.max(auto_corr) + 1e-8)
    lattice_bonus = np.std(rhythm_lattice) / (np.mean(rhythm_lattice) + 1e-8)
    return round(coherence * (1 - lattice_bonus), 4)  # Improved lattice penalty

# Structural invariance under +5 semitone shift (CQT preserves ratios)
def structural_invariance(y, sr, semitones=5):
    shift_factor = 2 ** (semitones / 12)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    cqt_orig = np.mean(np.abs(librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN, n_bins=CQT_BINS)), axis=1)
    cqt_shift = np.mean(np.abs(librosa.cqt(y_shifted, sr=sr, hop_length=HOP_LENGTH, fmin=CQT_FMIN * shift_factor, n_bins=CQT_BINS)), axis=1)
    invariance = 1 - np.mean(np.abs(cqt_orig - cqt_shift) / (np.max(cqt_orig) + 1e-8))
    return round(invariance, 4)

# Consonance bonus based on spectral centroid and rolloff
def consonance_bonus(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bonus = np.mean(centroid) / (np.mean(rolloff) + 1e-8)
    return round(bonus, 4)

# Main analysis function
def analyze_sound(y, sr, name):
    sound_type = classify_sound_type(y, sr)
    dom_freq = get_dominant_freq(y, sr) if sound_type != "noise" else 0.0
    harm_coh = harmonic_coherence(y, sr, dom_freq) if sound_type in ["pitched", "mixed"] else 0.0
    rhythm_coh = rhythm_coherence(y, sr)
    final_coh = max(harm_coh, rhythm_coh) if sound_type == "mixed" else (harm_coh if sound_type == "pitched" else rhythm_coh)
    invariance = structural_invariance(y, sr)
    bonus = consonance_bonus(y, sr)
    metrics = []
    if sound_type in ["pitched", "mixed"]: metrics.append("harmonic")
    if sound_type in ["noise", "mixed"] or rhythm_coh > 0.5: metrics.append("rhythm")
    metrics_str = "+".join(metrics) if metrics else "none"
    
    print(f"\n--- Analysis: {name} ---")
    print(f"Sound type: {sound_type}")
    if dom_freq > 0: print(f"Dominant: {dom_freq}")
    if harm_coh > 0: print(f"Harmonic Coherence: {harm_coh:.4f}")
    print(f"Rhythm Coherence: {rhythm_coh:.4f}")
    print(f"Final Coherence: {final_coh:.4f}")
    print(f"Structural Invariance(+5st): {invariance:.4f}")
    print(f"Consonance bonus: {bonus:.4f}")
    print(f"Applicable metrics: {metrics_str}")

# Broad sound analysis (improved for complex mixtures)
def analyze_broad_sound(file_path, sr):
    if not os.path.exists(file_path):
        return None
    y, _ = librosa.load(file_path, sr=sr)
    y_harm, y_perc = librosa.effects.hpss(y)
    analyze_sound(y_harm, sr, f"{file_path} (Harmonic Component)")
    analyze_sound(y_perc, sr, f"{file_path} (Percussive Component)")

# Main execution
if __name__ == "__main__":
    print("✅ Nightingale Mapping Rosetta Stone v16.5 – Refined Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling")
    
    # Test sounds
    y_a440 = generate_a440_tone(SR, DURATION)
    analyze_sound(y_a440, SR, "A440 Tone")
    
    y_noise = generate_white_noise(SR, DURATION)
    analyze_sound(y_noise, SR, "White Noise")
    
    y_beat = generate_rhythmic_beat(SR, DURATION)
    analyze_sound(y_beat, SR, "Rhythmic Beat")
    
    # Broad sound contrast
    print("\nBroad sound contrast (Beatles/orchestra) ready when files are uploaded to Replit root.")
    beatles_result = analyze_broad_sound('beatles.wav', SR)
    if beatles_result is None:
        print("Beatles file not found.")
    orchestra_result = analyze_broad_sound('orchestra.wav', SR)
    if orchestra_result is None:
        print("Orchestra file not found.")