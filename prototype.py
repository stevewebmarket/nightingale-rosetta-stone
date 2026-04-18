# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Refined Lattice Search and Finer CQT Hop
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file, sr=22050):
    y, sr = librosa.load(file, sr=sr)
    
    # Compute spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid < 1000:
        sound_type = 'low'
        print(f'  Detected low-centroid sound.')
        print(f'  Using low-sensitivity onset params.')
        onset_params = {'backtrack': False, 'delta': 0.1, 'wait': 1}
    elif mean_centroid < 4000:
        sound_type = 'mid'
        print(f'  Detected mid-centroid sound.')
        print(f'  Using mid-sensitivity onset params.')
        onset_params = {'backtrack': True, 'delta': 0.05, 'wait': 0}
    else:
        sound_type = 'high'
        print(f'  Detected high-centroid sound.')
        print(f'  Using high-sensitivity onset params.')
        onset_params = {'backtrack': True, 'delta': 0.02, 'wait': 0}
    
    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', **onset_params)
    print(f'  Detected onsets: {len(onsets)}')
    
    if len(onsets) < 2:
        print('  Insufficient onsets for rhythm analysis.')
        return
    
    # Compute IOIs and mean
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    rhythm_coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
    print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')
    
    # Improved rhythm lattice: search for best base period
    possible_bases = np.arange(0.005, 0.05, 0.001)
    best_coherence = 0
    best_base = 0.01
    onset_times = onsets
    for base in possible_bases:
        fits = 0
        for t in onset_times[1:]:
            multiple = round(t / base)
            if abs(t - multiple * base) < base * 0.1:  # 10% tolerance
                fits += 1
        coherence = fits / (len(onset_times) - 1)
        if coherence > best_coherence:
            best_coherence = coherence
            best_base = base
    print(f'  Rhythm lattice base: {best_base:.3f} s')
    print(f'  lattice coherence: {best_coherence:.2f}')
    
    # CQT with finer hop_length for better shift invariance
    hop_length = 256  # Reduced from default 512 for finer time resolution
    cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length, n_bins=672, bins_per_octave=96))
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')
    
    # Improved CQT shift invariance metric: average normalized difference on small shifts
    shift_diff = 0
    for shift in [1, 2]:  # Check invariance to small frame shifts
        cqt_shifted = np.roll(cqt, shift, axis=1)
        diff = np.mean(np.abs(cqt - cqt_shifted)) / np.mean(np.abs(cqt))
        shift_diff += diff
    invariance_metric = shift_diff / 2
    print(f'  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)')

if __name__ == '__main__':
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        print(f'Analysis for {file}:')
        analyze_audio(file)