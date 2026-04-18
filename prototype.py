# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Coherence and CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import mode

def analyze_audio(file, sr=22050):
    y, sr = librosa.load(file, sr=sr)
    
    # Spectral centroid for sound type classification
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1500:
        sound_type = 'low-centroid'
        sensitivity = 'low'
        onset_params = {'delta': 0.15, 'wait': 4, 'pre_max': 0.03, 'post_max': 0.03}
        hop_length = 512
    elif centroid < 4000:
        sound_type = 'mid-centroid'
        sensitivity = 'mid'
        onset_params = {'delta': 0.07, 'wait': 2, 'pre_max': 0.02, 'post_max': 0.02}
        hop_length = 256
    else:
        sound_type = 'high-centroid'
        sensitivity = 'high'
        onset_params = {'delta': 0.03, 'wait': 1, 'pre_max': 0.01, 'post_max': 0.01}
        hop_length = 128
    
    # Onset detection
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True, **onset_params)
    if len(onsets) < 2:
        print(f"  Insufficient onsets detected for {file}.")
        return
    
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    rhythm_coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 else 0
    
    # Improved rhythm lattice using tempo estimation for base period
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    lattice_base = 60 / tempo if tempo > 0 else mean_ioi
    
    # Lattice coherence: mean normalized residual to lattice points
    lattice_points = np.arange(0, onsets[-1] + lattice_base, lattice_base)
    residuals = [min(np.abs(t - lattice_points)) for t in onsets]
    mean_res = np.mean(residuals)
    lattice_coherence = max(0, 1 - (2 * mean_res / lattice_base))  # Normalized to [0,1]
    
    # CQT with adaptive hop_length
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'), n_bins=384, bins_per_octave=48)
    
    # Improved CQT shift invariance metric with normalization
    if cqt.shape[1] > 1:
        cqt_a = cqt[:, :-1]
        cqt_b = cqt[:, 1:]
        norm = np.sqrt(np.mean(np.abs(cqt_a)**2) * np.mean(np.abs(cqt_b)**2)) + 1e-8
        metric = np.mean(cqt_a * np.conj(cqt_b)) / norm
    else:
        metric = 0 + 0j
    
    # Print results
    print(f"Analysis for {file}:")
    print(f"  Detected {sound_type} sound.")
    print(f"  Using {sensitivity}-sensitivity onset params.")
    print(f"  Detected onsets: {len(onsets)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    real = metric.real
    imag = metric.imag
    sign = '+' if imag >= 0 else '-'
    print(f"  CQT shift invariance metric: {real:.2f}{sign}{abs(imag):.2f}j (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)