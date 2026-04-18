# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

def classify_sound(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1500:
        return 'low-centroid'
    elif centroid > 4000:
        return 'high-centroid'
    else:
        return 'mid-centroid'

def get_onset_params(sound_type, coherence=None):
    if sound_type == 'low-centroid':
        return {'delta': 0.08, 'wait': 4, 'backtrack': True}
    elif sound_type == 'high-centroid':
        return {'delta': 0.02, 'wait': 2, 'backtrack': False}
    else:
        params = {'delta': 0.04, 'wait': 3, 'backtrack': True}
        # Adaptive for broad sounds with low initial coherence
        if coherence is not None and coherence < 0.6:
            params['delta'] *= 0.75  # Lower threshold for irregular sounds
        return params

def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    cv = np.std(iois) / (np.mean(iois) + 1e-6)
    # Improved: clipped and scaled for better range
    return np.clip(1 - cv, 0, 1) ** 0.5  # Square root for less penalty on variability

def compute_lattice_base(y, sr, onsets):
    # Improved: Use tempogram-based tempo estimation for better lattice base
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    base = 60 / tempo if tempo > 0 else np.median(np.diff(onsets))
    # Refine for irregular rhythms
    if len(onsets) > 1:
        autocorr = librosa.autocorrelate(onset_env)
        peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
        if len(peaks) > 1:
            base = min(base, np.mean(np.diff(peaks)) * (1 / sr))  # Sample to time
    return base

def compute_lattice_coherence(times, base):
    if base == 0 or len(times) < 2:
        return 0.0
    # Improved: Use histogram entropy for coherence
    phases = (times % base) / base
    hist, _ = np.histogram(phases, bins=20, range=(0,1))
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log(hist + 1e-6))
    max_entropy = np.log(20)
    return 1 - (entropy / max_entropy)  # Higher when more concentrated

def compute_cqt_invariance(cqt):
    if cqt.shape[1] < 2:
        return 0.0
    # Improved: Normalize frames and average correlations over small shifts (1-3 frames)
    norms = np.linalg.norm(cqt, axis=0) + 1e-6
    cqt_norm = cqt / norms
    corrs = []
    for shift in [1, 2, 3]:
        for i in range(cqt_norm.shape[1] - shift):
            corr = np.dot(cqt_norm[:, i], cqt_norm[:, i + shift])
            corrs.append(corr)
    return np.mean(corrs) if corrs else 0.0

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

for file in files:
    print(f'Analysis for {file}:')
    y, sr = librosa.load(file, sr=22050)
    sound_type = classify_sound(y, sr)
    print(f'  Detected {sound_type} sound.')

    # Initial onset detection for coherence estimate
    initial_params = get_onset_params(sound_type)
    initial_onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **initial_params)
    initial_iois = np.diff(initial_onsets) if len(initial_onsets) > 1 else []
    initial_coherence = compute_rhythm_coherence(initial_iois)

    # Get possibly adapted params
    params = get_onset_params(sound_type, initial_coherence)
    print(f'  Using {sound_type.split("-")[0]}-sensitivity onset params.')

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **params)
    print(f'  Detected onsets: {len(onsets)}')

    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        rhythm_coh = compute_rhythm_coherence(iois)
        print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}')

        base = compute_lattice_base(y, sr, onsets)
        print(f'  Rhythm lattice base: {base:.3f} s')

        lattice_coh = compute_lattice_coherence(onsets, base)
        print(f'  lattice coherence: {lattice_coh:.2f}')

    # Improved CQT: Use higher resolution with more overlap for better invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=384, bins_per_octave=48, hop_length=256))
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')

    inv = compute_cqt_invariance(cqt)
    print(f'  CQT shift invariance metric: {inv:.2f} (higher is more invariant)')