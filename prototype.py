# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence and CQT Pitch Shift Invariance
# =============================================================================

import librosa
import numpy as np

def compute_rhythm_coherence(iois):
    if len(iois) == 0:
        return 0.0
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    cv = std_ioi / (mean_ioi + 1e-5)
    return 1 / (1 + cv)

def compute_rhythm_lattice(onsets):
    if len(onsets) < 2:
        return 0.0, 0.0
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    candidates = [mean_ioi / k for k in range(1, 21) if mean_ioi / k > 0.001]
    best_base = candidates[0]
    best_coherence = 0.0
    for base in candidates:
        errors = [min(t % base, base - t % base) for t in onsets]
        mean_error = np.mean(errors)
        normalized_error = mean_error / (base / 2 + 1e-5)
        coherence = max(0, 1 - normalized_error)
        if coherence > best_coherence:
            best_coherence = coherence
            best_base = base
    return best_base, best_coherence

def compute_cqt_invariance(cqt, bins_per_octave):
    cqt_mag = np.abs(cqt)
    norms = np.linalg.norm(cqt_mag, axis=0) + 1e-8
    norm_cqt = cqt_mag / norms
    # Shift by one octave (bins_per_octave bins)
    roll_cqt = np.roll(norm_cqt, bins_per_octave, axis=0)
    cos_sim_per_frame = np.sum(norm_cqt * roll_cqt, axis=0)
    metric = np.mean(cos_sim_per_frame)
    return metric

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fall back to synthetic test signals
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y = np.sin(2 * np.pi * 440 * t)  # Simple tone
        files = ['synthetic.wav']  # Placeholder, but analyze y directly
        # In practice, save and load, but for simplicity, process inline
    else:
        pass  # Use provided files

    for file in files:
        if file == 'synthetic.wav':
            # Use synthetic y from above
            pass
        else:
            y, sr = librosa.load(file, sr=22050)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_centroid = np.mean(centroid)

        if mean_centroid > 5000:
            sound_type = 'high-centroid sound.'
            fmin = librosa.note_to_hz('C4')
        elif mean_centroid < 1000:
            sound_type = 'low-centroid sound.'
            fmin = librosa.note_to_hz('C0')
        else:
            sound_type = 'mid-centroid sound.'
            fmin = librosa.note_to_hz('C1')

        # Onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onsets = librosa.frames_to_time(onset_frames, sr=sr)
        num_onsets = len(onsets)

        # IOIs and rhythm coherence
        iois = np.diff(onsets) if num_onsets > 1 else []
        mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
        rhythm_coherence = compute_rhythm_coherence(iois)

        # Rhythm lattice
        lattice_base, lattice_coherence = compute_rhythm_lattice(onsets)

        # CQT with adaptive params for broad handling
        bins_per_octave = 48
        n_bins = 384
        cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
        cqt_shape = cqt.shape

        # CQT shift invariance metric
        invariance_metric = compute_cqt_invariance(cqt, bins_per_octave)

        # Output
        print(f"Analysis for {file}:")
        print(f"  Detected {sound_type}")
        print(f"  Detected onsets: {num_onsets}")
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")
        print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
        print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")

if __name__ == "__main__":
    main()