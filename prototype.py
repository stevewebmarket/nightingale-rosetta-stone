# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, and Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
from scipy.signal import find_peaks

def analyze_file(file):
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1500:
        centroid_type = "low-centroid sound."
        fmin = 20.0
        onset_feature = librosa.feature.spectral_contrast  # Emphasize for low
    elif mean_centroid > 5000:
        centroid_type = "high-centroid sound."
        fmin = 200.0
        onset_feature = librosa.feature.mfcc  # For high, use MFCC for better transient capture
    else:
        centroid_type = "mid-centroid sound."
        fmin = None
        onset_feature = None
    print(f"  Detected {centroid_type}")
    
    # Onsets with adaptive handling
    if onset_feature is not None:
        # Custom onset strength for broad handling
        if centroid_type == "low-centroid sound.":
            o_env = np.mean(onset_feature(y=y, sr=sr)[:2, :], axis=0)  # Low bands
        elif centroid_type == "high-centroid sound.":
            o_env = np.mean(onset_feature(y=y, sr=sr), axis=0)  # Full for high
        else:
            o_env = librosa.onset.onset_strength(y=y, sr=sr)
    else:
        o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, units='time', backtrack=True)
    num_onsets = len(onset_times)
    print(f"  Detected onsets: {num_onsets}")
    
    if num_onsets > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        # Improved coherence: use Rayleigh test approximation for periodicity
        phases = np.cumsum(iois) % mean_ioi / mean_ioi
        rayleigh = np.abs(np.sum(np.exp(2j * np.pi * phases))) / len(phases)
        rhythm_coherence = rayleigh
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    else:
        print("  Insufficient onsets for IOI analysis.")
    
    # Improved rhythm lattice using tempogram for better periodicity detection
    tg = librosa.feature.tempogram(onset_envelope=o_env, sr=sr)
    tg_mean = np.mean(tg, axis=1)
    # Smooth for better peak
    tg_mean_smooth = np.convolve(tg_mean, np.ones(5)/5, mode='same')
    dominant_bin = np.argmax(tg_mean_smooth)
    bpm_bins = librosa.tempo_frequencies(len(tg_mean_smooth), sr=sr)
    dominant_bpm = bpm_bins[dominant_bin]
    if dominant_bpm > 0:
        lattice_base = 240 / dominant_bpm  # Assume 4-beat lattice for broader applicability
        # Improved coherence: peak prominence
        peaks, properties = find_peaks(tg_mean_smooth, prominence=0.1)
        if len(peaks) > 0 and dominant_bin in peaks:
            lattice_coherence = properties['prominences'][peaks.tolist().index(dominant_bin)] / np.max(tg_mean_smooth)
        else:
            lattice_coherence = 0.0
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")
    else:
        print("  No dominant rhythm lattice detected.")
    
    # Improved CQT: use hybrid for broad sounds, adaptive fmin
    cqt = librosa.hybrid_cqt(y=y, sr=sr, n_bins=384, fmin=fmin)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: log-magnitude, average corr over octave-relevant shifts
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    def shift_corr(cqt_mag, shift):
        shifted = np.roll(cqt_mag, shift=shift, axis=0)
        valid_len = min(cqt_mag.shape[1], shifted.shape[1])
        corr = np.corrcoef(cqt_mag[:, :valid_len].flatten(), shifted[:, :valid_len].flatten())[0, 1]
        return corr
    shifts = range(1, 13)  # Semitone to octave
    corrs = [shift_corr(cqt_mag, s) for s in shifts]
    metric = np.mean([c for c in corrs if not np.isnan(c)])
    print(f"  CQT shift invariance metric: {metric:.2f} (higher is more invariant)")

files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
for file in files:
    analyze_file(file)
    print()  # Blank line between analyses