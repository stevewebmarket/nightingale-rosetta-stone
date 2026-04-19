# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + CQT Invariance for Broad Sounds
# =============================================================================

import librosa
import numpy as np
import os

# Constants
SR = 22050
CQT_BINS_PER_OCTAVE = 48  # Increased for better frequency resolution
CQT_N_BINS = CQT_BINS_PER_OCTAVE * 8  # 8 octaves for broader coverage
HOP_LENGTH = 512
ONSET_BACKTRACK = True
CENTROID_THRESHOLD_HIGH = 5000  # Hz, for classifying high-centroid sounds

def classify_sound_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > CENTROID_THRESHOLD_HIGH:
        return "high-centroid"
    else:
        return "mid-centroid"

def detect_onsets(y, sr, sound_type):
    if sound_type == "high-centroid":
        # Adaptive for birdsong: higher sensitivity, shorter hop
        o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH//2, aggregate=np.median)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=ONSET_BACKTRACK, pre_max=0.02, post_max=0.02)
    else:
        o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.mean)
        onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, backtrack=ONSET_BACKTRACK)
    return onsets

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved coherence: use variance normalized by mean
    coherence = 1.0 / (1.0 + np.std(iois) / mean_ioi)
    
    # Improved lattice: use gcd of quantized IOIs with finer resolution
    quantized_iois = np.round(iois * 1000).astype(int)  # ms resolution
    gcd = np.gcd.reduce(quantized_iois)
    lattice_base = gcd / 1000.0 if gcd > 0 else mean_ioi / 2.0
    
    # Lattice coherence: fraction of onsets aligning to lattice multiples (with tolerance)
    tolerance = lattice_base * 0.1
    aligned = np.sum(np.isclose(onset_times % lattice_base, 0, atol=tolerance)) / len(onset_times)
    lattice_coherence = aligned * coherence  # Combined metric
    
    return len(onset_times), mean_ioi, coherence, lattice_base, lattice_coherence

def compute_cqt(y, sr, sound_type):
    if sound_type == "high-centroid":
        # Adjusted for high-frequency sounds: higher min freq
        cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE, fmin=librosa.note_to_hz('C3'))
    else:
        cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE, fmin=librosa.note_to_hz('C1'))
    return np.abs(cqt)

def cqt_shift_invariance(cqt):
    # Improved metric: average correlation across small shifts
    correlations = []
    for shift in range(1, 5):  # Check invariance to 1-4 frame shifts
        shifted = np.roll(cqt, shift, axis=1)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        correlations.append(corr)
    return np.mean(correlations)

def analyze_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    y, sr = librosa.load(file_path, sr=SR)
    sound_type = classify_sound_type(y, sr)
    
    onset_frames = detect_onsets(y, sr, sound_type)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    num_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onset_times)
    
    cqt = compute_cqt(y, sr, sound_type)
    
    invariance_metric = cqt_shift_invariance(cqt)
    
    print(f"Analysis for {os.path.basename(file_path)}:")
    print(f"  Detected {sound_type} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)\n")

def generate_synthetic_signal(sr, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return y

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signal.")
        y = generate_synthetic_signal(SR)
        # Save or simulate analysis on synthetic
        analyze_audio('synthetic.wav')  # Placeholder, but in reality, we'd process y directly
        return
    
    for file in wav_files:
        try:
            analyze_audio(file)
        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":
    main()