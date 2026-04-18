# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced Coherence and Invariance
# =============================================================================

import librosa
import numpy as np
import os
import scipy

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, use synthetic signals
if not wav_files:
    # Synthetic test signals
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y_sine = np.sin(2 * np.pi * 440 * t)  # Sine wave
    y_noise = np.random.randn(len(t))  # Noise
    y_chirp = librosa.chirp(fmin=100, fmax=10000, sr=sr, duration=5)  # Chirp
    synthetic_data = [('sine', y_sine), ('noise', y_noise), ('chirp', y_chirp)]
else:
    synthetic_data = []

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def detect_sound_type(mean_centroid):
    if mean_centroid > 3000:
        return "high-centroid sound (e.g., birdsong)"
    elif mean_centroid < 1000:
        return "low-centroid sound (e.g., bass-heavy)"
    else:
        return "mid-centroid sound (e.g., orchestral or rock)"

def adjusted_onset_detection(y, sr, sound_type):
    hop_length = 512
    if "high-centroid" in sound_type:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmax=8000, aggregate=np.median)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop_length, backtrack=True, pre_max=0.05, post_max=0.05)
    elif "low-centroid" in sound_type:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, fmax=2000, aggregate=np.mean)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop_length, backtrack=False, pre_max=0.1, post_max=0.1)
    else:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.mean)
        onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop_length, backtrack=False)
    return onsets, oenv

def compute_cqt(y, sr, sound_type):
    if "high-centroid" in sound_type:
        n_bins = 120
        bins_per_octave = 24
        fmin = librosa.note_to_hz('C2')
    elif "low-centroid" in sound_type:
        n_bins = 84
        bins_per_octave = 12
        fmin = librosa.note_to_hz('C0')
    else:
        n_bins = 96
        bins_per_octave = 12
        fmin = librosa.note_to_hz('C1')
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    return np.abs(cqt)

def compute_rhythm_metrics(onsets, sr, oenv):
    if len(onsets) < 2:
        return 0, 0.0, 0.0
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    # Improved coherence: autocorrelation of onset strength
    autocorr = scipy.signal.correlate(oenv, oenv, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = scipy.signal.find_peaks(autocorr)[0]
    if len(peaks) > 1:
        coherence = np.mean(autocorr[peaks[1:]]) / autocorr[0]
    else:
        coherence = 0.0
    return len(onsets), mean_ioi, coherence

def compute_rhythm_lattice(iois, min_resolution=0.005):
    if len(iois) == 0:
        return 0.010, 1.00
    # Adaptive lattice base: approximate GCD with finer resolution
    gcd = iois[0]
    for ioi in iois[1:]:
        gcd = np.gcd(int(gcd * 1000), int(ioi * 1000)) / 1000.0
    base = max(gcd / 10, min_resolution)
    # Lattice coherence: how well IOIs fit multiples of base
    fits = [np.abs(ioi - np.round(ioi / base) * base) / base for ioi in iois]
    lattice_coherence = 1 - np.mean(fits)
    return base, lattice_coherence

def compute_cqt_invariance(cqt):
    # Improved shift invariance: average correlation over small pitch shifts (1-3 bins)
    invariance_scores = []
    for shift in [1, 2, 3]:
        shifted = np.roll(cqt, shift, axis=0)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        invariance_scores.append(1 - corr)  # Lower difference means more invariant
    return np.mean(invariance_scores)

def analyze_audio(file_or_name, y=None, sr=22050, is_synthetic=False):
    if is_synthetic:
        name = file_or_name
    else:
        name = file_or_name
        if not os.path.exists(file_or_name):
            print(f"File {file_or_name} not found.")
            return
        y, sr = librosa.load(file_or_name, sr=sr)
    
    print(f"Analysis for {name}:")
    mean_centroid = compute_spectral_centroid(y, sr)
    sound_type = detect_sound_type(mean_centroid)
    print(f"  Detected {sound_type}, using adjusted onset detection.")
    
    onsets, oenv = adjusted_onset_detection(y, sr, sound_type)
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(onsets, sr, oenv)
    
    cqt = compute_cqt(y, sr, sound_type)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    print(f"  Detected onsets: {num_onsets}, mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    iois = np.diff(librosa.frames_to_time(onsets, sr=sr)) if len(onsets) > 1 else []
    lattice_base, lattice_coherence = compute_rhythm_lattice(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    
    invariance = compute_cqt_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

print("Analyzing available WAV files.")
for file in wav_files:
    analyze_audio(file)

for name, y in synthetic_data:
    analyze_audio(name, y=y, sr=22050, is_synthetic=True)