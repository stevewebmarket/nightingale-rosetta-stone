# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Optimized Lattice Coherence + Adaptive CQT Invariance
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    print(f'Analysis for {file}:')

    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)

    # Classify sound type and set parameters
    if mean_centroid < 1500:
        sound_type = 'low-centroid'
        sensitivity = 'low-sensitivity'
        delta = 0.2
        backtrack = False
        fmin = librosa.note_to_hz('C0')
    elif mean_centroid < 4000:
        sound_type = 'mid-centroid'
        sensitivity = 'mid-sensitivity'
        delta = 0.07
        backtrack = True
        fmin = librosa.note_to_hz('C1')
    else:
        sound_type = 'high-centroid'
        sensitivity = 'high-sensitivity'
        delta = 0.04
        backtrack = True
        fmin = librosa.note_to_hz('C2')

    print(f'  Detected {sound_type} sound.')
    print(f'  Using {sensitivity} onset params.')

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, delta=delta, backtrack=backtrack)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f'  Detected onsets: {len(onset_times)}')

    if len(onset_times) > 1:
        # Shift to start at 0
        onset_times -= onset_times[0]
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = np.exp(-cv)
        print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')

        # Find best rhythm lattice
        candidates = []
        for i in range(1, len(onset_times)):
            for k in range(1, 6):
                cand = onset_times[i] / k
                if cand > 0.01:
                    candidates.append(cand)
        candidates = np.unique(np.round(candidates, 3))

        best_coherence = 0
        best_base = mean_ioi / 2
        tol = 0.05  # 5% tolerance

        for base in candidates:
            hits = 0
            for t in onset_times:
                multiple = round(t / base)
                dist = abs(t - multiple * base)
                if dist < tol * base:
                    hits += 1
            coh = hits / len(onset_times)
            if coh > best_coherence:
                best_coherence = coh
                best_base = base

        print(f'  Rhythm lattice base: {best_base:.3f} s')
        print(f'  lattice coherence: {best_coherence:.2f}')
    else:
        print('  Insufficient onsets for rhythm analysis.')
        best_base = 0
        best_coherence = 0

    # Compute CQT
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=672, bins_per_octave=96)
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')

    # CQT shift invariance using lattice base
    base_time = best_base
    base_frames = librosa.time_to_frames(base_time, sr=sr)
    if base_frames > 0:
        similarities = []
        cqt_mag = np.abs(cqt)
        flat_orig = cqt_mag.flatten()
        norm_orig = np.linalg.norm(flat_orig)
        for m in range(1, 4):
            shift_f = m * base_frames
            if shift_f >= cqt.shape[1]:
                break
            shifted = np.roll(cqt_mag, shift_f, axis=1)
            flat_shift = shifted.flatten()
            norm_shift = np.linalg.norm(flat_shift)
            if norm_orig > 0 and norm_shift > 0:
                sim = np.dot(flat_orig, flat_shift) / (norm_orig * norm_shift)
                similarities.append(sim)
        if similarities:
            metric = np.mean(similarities)
        else:
            metric = 0
    else:
        metric = 0
    print(f'  CQT shift invariance metric: {metric:.2f} (higher is more invariant)')

if __name__ == '__main__':
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)