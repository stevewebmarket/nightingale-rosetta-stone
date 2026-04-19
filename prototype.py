# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice, Improved Coherence and Invariance
# =============================================================================

import librosa
import numpy as np
import math
import os

def analyze_audio(file):
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)

    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        print("  Detected high-centroid sound.")
    else:
        print("  Detected mid-centroid sound.")

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    num_onsets = len(onset_times)
    print(f"  Detected onsets: {num_onsets}")

    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = 100 / (1 + cv * 10)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        mean_ioi = 0.0
        rhythm_coherence = 0.0
        print("  mean IOI: 0.00 s, rhythm coherence: 0.00")

    # Improved rhythm lattice: adaptive based on estimated tempo
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_strength, sr=sr)[0]
    beat_dur = 60 / tempo if tempo > 0 else mean_ioi
    lattice_base = beat_dur / 16  # Assuming 16th note subdivisions for lattice
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")

    # Improved lattice coherence: alignment of onsets to lattice points
    duration = librosa.get_duration(y=y, sr=sr)
    lattice_points = np.arange(0, duration + lattice_base, lattice_base)
    coherence_sum = 0.0
    threshold = lattice_base * 0.1  # 10% tolerance for alignment
    for ot in onset_times:
        dists = np.abs(lattice_points - ot)
        min_dist = np.min(dists)
        if min_dist < threshold:
            coherence_sum += 1 - (min_dist / threshold)
    lattice_coherence = coherence_sum / num_onsets if num_onsets > 0 else 0.0
    print(f"  lattice coherence: {lattice_coherence:.2f}")

    # Improved CQT for better invariance and broad sound handling
    # Adaptive fmin based on centroid for broader frequency handling
    fmin = 32.7 if mean_centroid < 5000 else 100.0  # Higher fmin for high-centroid sounds
    cqt = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=384, bins_per_octave=48)))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

    # Improved shift invariance metric: average correlation of normalized consecutive frames (time-shift invariance)
    def normalize_frame(frame):
        norm = np.linalg.norm(frame)
        return frame / norm if norm > 0 else frame

    inv = 0.0
    count = 0
    for t in range(cqt.shape[1] - 1):
        f1 = normalize_frame(cqt[:, t])
        f2 = normalize_frame(cqt[:, t + 1])
        corr = np.dot(f1, f2)
        inv += corr
        count += 1
    invariance = inv / count if count > 0 else 0.0
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print("")

def generate_synthetic_signal(sr=22050, duration=5.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 440.0
    y = np.sin(2 * np.pi * freq * t)
    return y

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files or not all(os.path.exists(f) for f in files):
        print("No WAV files found. Falling back to synthetic test signals.")
        # Synthetic signals for fallback
        sr = 22050
        y_bird = generate_synthetic_signal(sr, 5.0) + np.random.normal(0, 0.1, int(sr * 5))  # Noisy for birdsong-like
        librosa.output.write_wav('synthetic_birdsong.wav', y_bird, sr)
        y_orch = np.concatenate([generate_synthetic_signal(sr, 1.0, f) for f in [261.63, 329.63, 392.00]])  # Chords
        librosa.output.write_wav('synthetic_orchestra.wav', y_orch, sr)
        y_rock = generate_synthetic_signal(sr, 5.0, 440) * np.linspace(1, 0, int(sr * 5))  # Decaying
        librosa.output.write_wav('synthetic_rock.wav', y_rock, sr)
        files = ['synthetic_birdsong.wav', 'synthetic_orchestra.wav', 'synthetic_rock.wav']

    for file in files:
        analyze_audio(file)