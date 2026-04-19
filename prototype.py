# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Optimization + CQT Refinements
# =============================================================================

import librosa
import numpy as np

def analyze_audio(filename):
    y, sr = librosa.load(filename, sr=22050)

    # Spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)

    if mean_centroid > 5000:
        sound_type = "high-centroid sound."
        fmin = librosa.note_to_hz('C3')  # Adjust for high-centroid sounds
    elif mean_centroid > 2000:
        sound_type = "mid-centroid sound."
        fmin = librosa.note_to_hz('C1')
    else:
        sound_type = "low-centroid sound."
        fmin = librosa.note_to_hz('C1')

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    num_onsets = len(onset_frames)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv_ioi = std_ioi / mean_ioi if mean_ioi > 0 else 1
        rhythm_coherence = 100 * np.exp(-2 * cv_ioi)
    else:
        mean_ioi = 0
        cv_ioi = 1
        rhythm_coherence = 0

    # Rhythm lattice base with improved optimization
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    lattice_base = 60 / tempo if tempo > 0 else mean_ioi

    # Optimize lattice base for higher coherence
    lattice_coherence = 0
    if num_onsets > 1 and mean_ioi > 0:
        candidates = np.linspace(mean_ioi / 4, mean_ioi, 30)
        best_coh = 0
        best_base = lattice_base
        for base in candidates:
            lattice_points = np.arange(0, onset_times[-1] + base, base)
            hits = 0
            for ot in onset_times:
                dist = np.min(np.abs(ot - lattice_points))
                if dist < base * 0.1:
                    hits += 1
            coh = hits / num_onsets
            if coh > best_coh:
                best_coh = coh
                best_base = base
        lattice_base = best_base
        lattice_coherence = best_coh

    # CQT with improvements for invariance and broad handling
    hop_length = 256  # Smaller hop for better time resolution and invariance
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=384, bins_per_octave=48))
    cqt_db = librosa.amplitude_to_db(cqt)  # Log amplitude for better invariance

    cqt_shape = cqt_db.shape

    # CQT shift invariance metric (mean cosine similarity between consecutive frames)
    norms = np.linalg.norm(cqt_db, axis=0) + 1e-8
    norm_frames = cqt_db / norms
    similarities = [np.dot(norm_frames[:, i], norm_frames[:, i + 1]) for i in range(cqt_shape[1] - 1)]
    invariance = np.mean(similarities) if similarities else 0

    # Print analysis
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type}")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print("")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for filename in files:
    analyze_audio(filename)