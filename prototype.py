# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np

def compute_lattice(onset_times):
    if len(onset_times) < 2:
        return 0, 0
    onset_times = onset_times - onset_times[0]
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    possible_bases = np.linspace(0.05, 1.0, 200)
    best_coherence = 0
    best_base = 0
    tol = 0.03  # 30ms tolerance
    for base in possible_bases:
        if base == 0:
            continue
        multiples = np.round(onset_times / base)
        residuals = np.abs(onset_times - multiples * base)
        fit = np.mean(residuals < tol)
        if fit > best_coherence:
            best_coherence = fit
            best_base = base
    return best_base, best_coherence

def analyze_audio(file):
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_cent = np.mean(centroid)
    
    if mean_cent > 5000:
        print("  Detected high-centroid sound (e.g., birdsong), using adjusted onset detection.")
        type_ = 'high'
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, fmax=8000, n_mels=256)
        onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
        fmin = librosa.note_to_hz('C4')
    else:
        print("  Detected mid-centroid sound (e.g., orchestral or rock), using adjusted onset detection.")
        type_ = 'mid'
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(beat_frames, sr=sr)
        fmin = librosa.note_to_hz('C1')
    
    cqt = np.abs(librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=96, bins_per_octave=12))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance with normalization
    eps = 1e-6
    norms = np.linalg.norm(cqt, axis=0) + eps
    cqt_norm = cqt / norms[:, np.newaxis].T
    diffs = cqt_norm[:, :-1] - cqt_norm[:, 1:]
    dists = np.linalg.norm(diffs, axis=0)
    shift_metric = np.mean(dists)
    
    n_onsets = len(onset_times)
    if n_onsets < 2:
        print("  Too few onsets detected.")
        return
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved rhythm coherence using coefficient of variation
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else np.inf
    rhythm_coherence = 1 / (1 + cv) if np.isfinite(cv) else 0
    
    lattice_base, lattice_coherence = compute_lattice(onset_times)
    
    print(f"  Detected onsets: {n_onsets}, mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shift invariance metric: {shift_metric:.2f} (lower is more invariant)")

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)