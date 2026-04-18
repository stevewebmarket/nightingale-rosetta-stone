# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np

def get_centroid_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 4000:
        return 'high'
    elif centroid < 1000:
        return 'low'
    else:
        return 'mid'

def detect_onsets(y, sr, c_type):
    if c_type == 'high':
        hop_length = 256
        pre_max = 0.03
        post_max = 0.03
    elif c_type == 'low':
        hop_length = 1024
        pre_max = 0.1
        post_max = 0.1
    else:
        hop_length = 512
        pre_max = 0.05
        post_max = 0.05
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, backtrack=True, pre_max=pre_max, post_max=post_max)
    return onsets, o_env, hop_length

def analyze_rhythm(onset_frames, o_env, hop_length, sr):
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    if len(onset_times) < 2:
        return 0, 0, 0, 0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 1 / (1 + cv)
    tempo = librosa.beat.tempo(onset_envelope=o_env, sr=sr, hop_length=hop_length)[0]
    beat_duration = 60 / tempo
    lattice_base = beat_duration / 4
    phases = onset_times % lattice_base
    phase = np.median(phases)
    adjusted_times = onset_times - phase
    mods = adjusted_times % lattice_base
    tol = lattice_base * 0.1
    aligned = np.mean((mods < tol) | (lattice_base - mods < tol))
    return mean_ioi, rhythm_coherence, lattice_base, aligned

def compute_cqt_invariance(y, sr):
    hop_length = 256
    cqt_orig = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48))
    metrics = []
    for shift in range(1, 11):
        y_shift = np.roll(y, shift)
        cqt_shift = np.abs(librosa.cqt(y_shift, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48))
        min_frames = min(cqt_orig.shape[1], cqt_shift.shape[1])
        c1 = cqt_orig[:, :min_frames]
        c2 = cqt_shift[:, :min_frames]
        corrs = [np.corrcoef(c1[i], c2[i])[0,1] for i in range(c1.shape[0]) if not np.isnan(np.corrcoef(c1[i], c2[i])[0,1])]
        metrics.append(np.mean(corrs))
    return np.mean(metrics)

def analyze_file(file):
    y, sr = librosa.load(file, sr=22050)
    c_type = get_centroid_type(y, sr)
    print(f'  Detected {c_type}-centroid sound.')
    if c_type == 'high':
        print('  Using high-sensitivity onset params.')
    elif c_type == 'low':
        print('  Using low-sensitivity onset params.')
    else:
        print('  Using mid-sensitivity onset params.')
    onsets, o_env, hop = detect_onsets(y, sr, c_type)
    print(f'  Detected onsets: {len(onsets)}')
    mean_ioi, rhythm_coh, lattice_base, lattice_coh = analyze_rhythm(onsets, o_env, hop, sr)
    print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coh:.2f}')
    print(f'  Rhythm lattice base: {lattice_base:.3f} s')
    print(f'  lattice coherence: {lattice_coh:.2f}')
    cqt = librosa.cqt(y, sr=sr, n_bins=384, bins_per_octave=48)
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')
    inv_metric = compute_cqt_invariance(y, sr)
    print(f'  CQT shift invariance metric: {inv_metric:.2f} (higher is more invariant)')

if __name__ == '__main__':
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        print(f'Analysis for {file}:')
        analyze_file(file)