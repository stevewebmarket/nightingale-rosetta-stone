# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # Spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        sound_type = "high-centroid sound"
    elif mean_centroid > 2000:
        sound_type = "mid-centroid sound"
    else:
        sound_type = "low-centroid sound"
    print(f"  Detected {sound_type}.")

    # Onset detection, adaptive to sound type
    if "high-centroid" in sound_type:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, delta=0.1)
    else:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    print(f"  Detected onsets: {len(onsets)}")

    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        # Improved coherence: coefficient of variation inverted
        cv = np.std(iois) / mean_ioi
        rhythm_coherence = 1 / (cv + 1)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  Insufficient onsets for IOI calculation.")
        return

    # Improved rhythm lattice: base derived from approximate GCD of IOIs
    iois_rounded = np.round(iois * 1000).astype(int)  # to ms for gcd
    gcd = np.gcd.reduce(iois_rounded)
    lattice_base = gcd / 1000.0 if gcd > 0 else 0.023
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")

    # Improved lattice coherence: alignment error to lattice
    lattice_points = np.arange(0, len(y)/sr, lattice_base)
    snapped = [lattice_points[np.argmin(np.abs(lattice_points - o))] for o in onsets]
    errors = np.abs(np.array(snapped) - onsets)
    lattice_coherence = 1 - (np.mean(errors) / lattice_base) if lattice_base > 0 else 0
    lattice_coherence = max(0, min(1, lattice_coherence))
    print(f"  lattice coherence: {lattice_coherence:.2f}")

    # CQT with high resolution for better invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

    # Improved CQT shift invariance metric: correlation with octave-shifted signal's CQT
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=12)
    cqt_shifted = librosa.cqt(y_shifted, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.0)
    bpo = 48
    shifted_cqt = np.roll(np.abs(cqt), -bpo, axis=0)
    # Trim to min time frames
    min_frames = min(shifted_cqt.shape[1], cqt_shifted.shape[1])
    corr = np.corrcoef(np.abs(shifted_cqt[:, :min_frames]).flatten(), np.abs(cqt_shifted[:, :min_frames]).flatten())[0, 1]
    print(f"  CQT shift invariance metric: {corr:.2f} (higher is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
if not files:
    # Fallback to synthetic signals if no files
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y_sine = np.sin(2 * np.pi * 440 * t)
    y_chirp = librosa.chirp(fmin=100, fmax=10000, sr=sr, duration=5)
    np.save('synthetic_sine.npy', y_sine)  # But since we load wav, this is placeholder
    print("No WAV files; using synthetic signals (not implemented for WAV load).")
else:
    for file in files:
        print(f"Analysis for {file}:")
        analyze_audio(file)