# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

def analyze(y, sr, name):
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_cent = np.mean(centroid)
    if mean_cent > 3000:
        cent_type = "high-centroid"
    else:
        cent_type = "mid-centroid"

    # Onsets with adaptive parameters
    hop_length_onset = 256 if cent_type == "high-centroid" else 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length_onset)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, delta=0.1 if cent_type == "high-centroid" else 0.05)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length_onset)
    num_onsets = len(onset_times)

    # IOIs and rhythm coherence
    if num_onsets > 1:
        iois = np.diff(onset_times)
        iois = iois[iois > 0.01]  # Filter very small IOIs for better coherence
        if len(iois) > 0:
            mean_ioi = np.mean(iois)
            cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
            rhythm_coherence = 1 / (1 + cv)
        else:
            mean_ioi = 0
            rhythm_coherence = 0
    else:
        mean_ioi = 0
        rhythm_coherence = 0

    # Improved rhythm lattice
    if rhythm_coherence > 0.4 and len(onset_env) > 0:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        if tempo > 0:
            beat_period = 60 / tempo
            lattice_base = beat_period / 8  # Smaller unit for lattice
        else:
            lattice_base = mean_ioi / 10 if mean_ioi > 0 else 0.01
    else:
        lattice_base = np.median(iois) / 10 if len(iois) > 0 else 0.01

    # Lattice coherence
    if lattice_base > 0 and num_onsets > 0:
        lattice_points = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
        aligned = 0
        tol = lattice_base * 0.1
        for ot in onset_times:
            if np.min(np.abs(lattice_points - ot)) < tol:
                aligned += 1
        lattice_coherence = aligned / num_onsets
    else:
        lattice_coherence = 0

    # CQT with adaptive parameters for better invariance
    hop_length = 256 if cent_type == "high-centroid" else 512
    filter_scale = 1.5 if cent_type == "high-centroid" else 1.0
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48, filter_scale=filter_scale))
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]

    # CQT shift invariance metric
    shift_sec = 0.01
    shift_samples = int(shift_sec * sr)
    y_shifted = np.roll(y, shift_samples)
    cqt_shifted = np.abs(librosa.cqt(y=y_shifted, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48, filter_scale=filter_scale))
    min_frames = min(cqt.shape[1], cqt_shifted.shape[1])
    sims = []
    for b in range(n_bins):
        v1 = cqt[b, :min_frames]
        v2 = cqt_shifted[b, :min_frames]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            sim = np.dot(v1, v2) / (norm1 * norm2)
            sims.append(sim)
    invariance = np.mean(sims) if sims else 0

    # Print analysis
    print(f"Analysis for {name}:")
    print(f"  Detected {cent_type} sound.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    files = [f for f in wav_files if os.path.exists(f)]
    if not files:
        print("No WAV files found, using synthetic test signal.")
        sr = 22050
        duration = 10
        t = np.arange(duration * sr) / sr
        y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.3 * np.random.randn(len(t))
        analyze(y, sr, "synthetic_signal.wav")
    else:
        for file in files:
            y, sr = librosa.load(file, sr=22050)
            analyze(y, sr, file)
            print()