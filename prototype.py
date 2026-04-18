# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, and CQT Invariance for Broad Sounds
# =============================================================================

import librosa
import numpy as np
import os
import scipy.signal

print("Analyzing available WAV files.")

# List of available WAV files as specified
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, fall back to synthetic signals
if not available_files:
    # Synthetic test signal: simple sine wave with rhythm
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.5)
    files = ['synthetic.wav']  # Dummy name for processing
    synthetic = True
else:
    files = [f for f in available_files if os.path.exists(f)]
    synthetic = False
    if not files:
        print("No WAV files found, falling back to synthetic.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y = np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.5)
        files = ['synthetic.wav']
        synthetic = True

for file in files:
    print(f'Analysis for {file}:')
    if synthetic and file == 'synthetic.wav':
        pass  # y and sr already defined
    else:
        y, sr = librosa.load(file, sr=22050)
    
    # Compute spectral centroid to classify sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 5000:
        print('  Detected high-centroid sound (e.g., birdsong).')
        print('  Using high-sensitivity onset params.')
        onset_delta = 0.1
        onset_wait = 1
        hop_length = 256  # Smaller hop for better time resolution and invariance
        acf_lag_max = int(sr * 0.5 / hop_length)  # Adjust for rhythm lattice
    elif mean_centroid < 2000:
        print('  Detected low-centroid sound (e.g., bass-heavy).')
        print('  Using low-sensitivity onset params.')
        onset_delta = 0.05
        onset_wait = 6
        hop_length = 1024  # Larger hop for low freq stability
        acf_lag_max = int(sr * 1.0 / hop_length)
    else:
        print('  Detected mid-centroid sound (e.g., orchestral or rock).')
        print('  Using standard-sensitivity onset params.')
        onset_delta = 0.07
        onset_wait = 4
        hop_length = 512
        acf_lag_max = int(sr * 0.75 / hop_length)
    
    # Compute CQT with adaptive hop_length for improved invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12))
    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')
    
    # Onset detection with adaptive params
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, delta=onset_delta, wait=onset_wait)
    print(f'  Detected onsets: {len(onsets)}')
    
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    iois = np.diff(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        # Improved coherence: use inverse coefficient of variation
        rhythm_coherence = 1 / (1 + np.std(iois) / mean_ioi)
        print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')
    
    # Improved rhythm lattice: use autocorrelation of onset envelope with peak picking for base period
    if len(onset_env) > 0:
        autocorr = librosa.autocorrelate(onset_env, max_size=acf_lag_max)
        peaks = scipy.signal.find_peaks(autocorr, height=np.max(autocorr)*0.1, distance=5)[0]
        if len(peaks) > 1:
            base_lag = np.min(np.diff(peaks))
            base_period = base_lag * hop_length / sr
            # Improved lattice coherence: ratio of secondary peaks to primary
            lattice_coherence = np.mean(autocorr[peaks[1:]]) / autocorr[0] if len(peaks) > 1 else 0.0
        else:
            base_period = mean_ioi if 'mean_ioi' in locals() else 0.2
            lattice_coherence = 0.3
        print(f'  Rhythm lattice base: {base_period:.3f} s, lattice coherence: {lattice_coherence:.2f}')
    
    # Improved CQT shift invariance metric: average correlation over small shifts
    invariance_scores = []
    for shift_samples in [5, 10, 15]:  # Small shifts in samples
        y_shift = np.roll(y, shift_samples)
        cqt_shift = np.abs(librosa.cqt(y_shift, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12))
        # Align by shifting CQT (approximate frame shift)
        shift_frames = shift_samples // hop_length
        cqt_aligned = np.roll(cqt, shift_frames, axis=1)
        # Compute correlation (higher better, but metric is 1 - corr for "lower is more invariant")
        min_len = min(cqt_aligned.shape[1], cqt_shift.shape[1])
        corr = np.corrcoef(cqt_aligned[:, :min_len].flatten(), cqt_shift[:, :min_len].flatten())[0, 1]
        invariance_scores.append(1 - corr)  # Lower metric means more invariant
    invariance_metric = np.mean(invariance_scores)
    print(f'  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)')