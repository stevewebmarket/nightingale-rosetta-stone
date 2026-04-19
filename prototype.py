# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + CQT Invariance Enhancements
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, fall back to synthetic signals
if not wav_files:
    # Synthetic test signals
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y_sine = np.sin(2 * np.pi * 440 * t)  # A4 sine wave
    y_chirp = librosa.chirp(fmin=100, fmax=10000, sr=sr, duration=5)
    y_noise = np.random.randn(5 * sr)
    test_signals = [('sine', y_sine), ('chirp', y_chirp), ('noise', y_noise)]
else:
    test_signals = None

def detect_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return "high-centroid sound"
    elif mean_centroid > 2000:
        return "mid-centroid sound"
    else:
        return "low-centroid sound"

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    # Improved coherence: coefficient of variation (lower is more coherent)
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    coherence = 1 / (1 + cv)  # Normalize to [0,1], higher better
    return len(onset_times), mean_ioi, coherence

def adaptive_rhythm_lattice(onset_times, sr):
    if len(onset_times) < 2:
        return 0.05, 0.0
    iois = np.diff(onset_times)
    # Adaptive base: median IOI divided by a factor, clamped
    base = np.median(iois) / 4
    base = max(0.01, min(0.1, base))  # Clamp between 0.01s and 0.1s
    # Lattice: multiples of base up to max IOI
    lattice = np.arange(0, np.max(onset_times) + base, base)
    # Coherence: fraction of onsets within 0.01s tolerance of lattice points
    hits = 0
    tol = 0.01
    for ot in onset_times:
        if np.min(np.abs(lattice - ot)) <= tol:
            hits += 1
    coherence = hits / len(onset_times) if len(onset_times) > 0 else 0
    return base, coherence

def improved_cqt(y, sr):
    # Enhanced CQT with adjusted parameters for better invariance
    # Increased bins_per_octave for finer resolution, adjusted hop_length
    hop_length = 256  # Smaller hop for better time resolution
    return librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'),
                        n_bins=168, bins_per_octave=24)  # 24 bpo for better shift invariance

def cqt_shift_invariance(cqt):
    # Improved metric: autocorrelation across time shifts
    # Normalize CQT magnitude
    cqt_mag = np.abs(cqt)
    cqt_norm = cqt_mag / (np.max(cqt_mag) + 1e-6)
    # Compute average correlation with shifted versions (1-5 frames)
    corrs = []
    for shift in range(1, 6):
        corr = np.corrcoef(cqt_norm[:, :-shift].flatten(), cqt_norm[:, shift:].flatten())[0, 1]
        corrs.append(corr)
    return np.mean(corrs)

def analyze_audio(file_or_signal, sr=22050, is_file=True):
    if is_file:
        if not os.path.exists(file_or_signal):
            print(f"File {file_or_signal} not found.")
            return
        y, sr = librosa.load(file_or_signal, sr=sr)
        name = file_or_signal
    else:
        name, y = file_or_signal
    print(f"Analysis for {name}:")
    sound_type = detect_sound_type(y, sr)
    print(f"  Detected {sound_type}.")
    
    # Onset detection with backtrack for better accuracy
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(onset_times)
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Adaptive rhythm lattice
    lattice_base, lattice_coherence = adaptive_rhythm_lattice(onset_times, sr)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Improved CQT
    cqt = improved_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Invariance metric
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    if wav_files:
        for file in wav_files:
            analyze_audio(file)
    elif test_signals:
        for signal in test_signals:
            analyze_audio(signal, is_file=False)
    else:
        print("No audio files or test signals available.")