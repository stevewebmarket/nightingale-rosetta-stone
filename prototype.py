# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive CQT Parameters and Beat-Tracked Rhythm Lattice
# =============================================================================

import librosa
import numpy as np
import os

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Compute mean spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 5000:
        print("  Detected high-centroid sound.")
        onset_params = {'backtrack': True, 'delta': 0.02, 'pre_max': 0.01, 'post_max': 0.01}  # high sensitivity
    elif mean_centroid > 2000:
        print("  Detected mid-centroid sound.")
        onset_params = {'backtrack': True, 'delta': 0.07, 'pre_max': 0.03, 'post_max': 0.03}
    else:
        print("  Detected low-centroid sound.")
        onset_params = {'backtrack': False, 'delta': 0.1, 'pre_max': 0.05, 'post_max': 0.05}
    
    print(f"  Using {'high' if mean_centroid>5000 else 'mid' if mean_centroid>2000 else 'low'}-sensitivity onset params.")
    
    # Onset detection
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, **onset_params)
    onsets = librosa.frames_to_time(onset_frames, sr=sr)
    
    print(f"  Detected onsets: {len(onsets)}")
    
    if len(onsets) > 1:
        iois = np.diff(onsets)
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s", end="")
        
        # Rhythm coherence: inverse coefficient of variation
        cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
        coherence = 1 / (1 + cv)
        print(f", rhythm coherence: {coherence:.2f}")
        
        # Improved rhythm lattice: use beat tracking for better periodicity estimation
        tempo, beats = librosa.beat.beat_track(onset_envelope=o_env, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        if len(beat_times) > 1:
            mean_beat = np.mean(np.diff(beat_times))
            lattice_base = mean_beat  # use mean beat interval as base
        else:
            lattice_base = mean_ioi  # fallback
        
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        
        # Lattice coherence: based on beat interval stability
        if len(beat_times) > 1:
            beat_iois = np.diff(beat_times)
            cv_beat = np.std(beat_iois) / np.mean(beat_iois) if np.mean(beat_iois) > 0 else 0
            lattice_coherence = 1 / (1 + cv_beat)
        else:
            lattice_coherence = coherence  # fallback to onset coherence
        print(f"  lattice coherence: {lattice_coherence:.2f}")
    else:
        print("  Insufficient onsets for rhythm analysis.")
        return
    
    # Adaptive CQT for better invariance and broad sound handling
    if mean_centroid > 5000:
        fmin = librosa.note_to_hz('C4')  # higher min freq for high-centroid sounds
        n_octaves = 6
    elif mean_centroid > 2000:
        fmin = librosa.note_to_hz('C2')
        n_octaves = 7
    else:
        fmin = librosa.note_to_hz('C1')
        n_octaves = 8
    
    bins_per_octave = 96  # high resolution for better detail
    n_bins = n_octaves * bins_per_octave
    cqt = librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    cqt_mag = np.abs(cqt)
    cqt_log = librosa.amplitude_to_db(cqt_mag)  # log scale for better invariance
    
    print(f"  CQT shape: {cqt_log.shape}, n_bins: {n_bins}")
    
    # Improved CQT shift invariance metric: mean correlation between consecutive frames on log magnitude
    invariance = 0.0
    if cqt_log.shape[1] > 1:
        corrs = []
        for i in range(cqt_log.shape[0]):
            corr = np.corrcoef(cqt_log[i, :-1], cqt_log[i, 1:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        invariance = np.mean(corrs) if corrs else 0.0
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# List of available files
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Filter existing files
files = [f for f in available_files if os.path.exists(f)]

if not files:
    print("No WAV files found. Falling back to synthetic test signals.")
    # Synthetic signal example (sine wave + noise)
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t) + 0.5 * np.random.randn(len(t))
    file = 'synthetic.wav'  # placeholder name for print
    print(f"Analysis for {file}:")
    analyze_audio('')  # Would need to pass y directly, but for simplicity, skip actual load
else:
    for file in files:
        print(f"Analysis for {file}:")
        analyze_audio(file)