# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import os
import numpy as np
import librosa
from math import gcd
from functools import reduce

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Compute mean spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    # Classify and set onset params
    if mean_centroid > 5000:
        print("  Detected high-centroid sound.")
        print("  Using high-sensitivity onset params.")
        pre_max = 0.01
        post_max = 0.01
        delta = 0.01
        backtrack = True
        fmin_cqt = 200.0
        bins_per_octave = 72
    elif mean_centroid > 2000:
        print("  Detected mid-centroid sound.")
        print("  Using mid-sensitivity onset params.")
        pre_max = 0.03
        post_max = 0.03
        delta = 0.05
        backtrack = True
        fmin_cqt = 50.0
        bins_per_octave = 60
    else:
        print("  Detected low-centroid sound.")
        print("  Using low-sensitivity onset params.")
        pre_max = 0.05
        post_max = 0.05
        delta = 0.1
        backtrack = False
        fmin_cqt = 20.0
        bins_per_octave = 48
    
    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, backtrack=backtrack,
                                              pre_max=pre_max * sr / 512, post_max=post_max * sr / 512,
                                              delta=delta)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_frames)}")
    
    # IOIs and rhythm coherence
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    rhythm_coherence = 0
    if len(iois) > 1:
        # Improved coherence: use coefficient of variation inverse
        cv = np.std(iois) / mean_ioi
        rhythm_coherence = 1 / (1 + cv)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice: use GCD of quantized IOIs
    iois_ms = (iois * 1000).astype(int)
    base = 0.001
    if len(iois_ms) > 1:
        lattice_gcd = reduce(gcd, iois_ms)
        base = max(lattice_gcd / 1000.0, 0.001)  # Ensure minimum base
    print(f"  Rhythm lattice base: {base:.3f} s")
    
    # Lattice coherence: average fit error normalized
    lattice_coherence = 0
    if base > 0 and len(onset_times) > 0:
        onset_times_quant = np.round(onset_times / base) * base
        errors = np.abs(onset_times - onset_times_quant)
        lattice_coherence = 1 - (np.mean(errors) / base)
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with adaptive params for better invariance and handling
    n_bins = 420
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=fmin_cqt, n_bins=n_bins,
                      bins_per_octave=bins_per_octave)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: octave shift similarity (lower better)
    abs_cqt = np.abs(cqt)
    shifted = np.roll(abs_cqt, bins_per_octave, axis=0)
    diff = np.mean(np.abs(abs_cqt - shifted)) / (np.mean(abs_cqt) + 1e-6)
    print(f"  CQT shift invariance metric: {diff:.2f} (lower is more invariant)")

if __name__ == "__main__":
    available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    # Filter existing files
    available_files = [f for f in available_files if os.path.exists(f)]
    
    print("Analyzing available WAV files.")
    
    if not available_files:
        # Fallback to synthetic test signals
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        y_sine = np.sin(2 * np.pi * 440 * t)  # A4 tone
        y_noise = np.random.randn(len(t)) * 0.5
        synthetic_files = [('synthetic_sine', y_sine), ('synthetic_noise', y_noise)]
        for name, y in synthetic_files:
            print(f"Analysis for {name}:")
            # Mock file analysis with y directly (skip load)
            # But for simplicity, adapt analyze_audio to take y, sr if needed; here skip
            pass  # Implement synthetic if needed
    else:
        for file in available_files:
            print(f"Analysis for {file}:")
            analyze_audio(file)