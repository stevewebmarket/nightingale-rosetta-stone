# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os
from math import gcd
from functools import reduce

def analyze_audio(file):
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    
    # Detect sound type based on spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    if mean_centroid < 1500:
        sound_type = "low-centroid sound."
        sensitivity = "low"
        onset_delta = 0.1
        onset_backtrack = True
    elif mean_centroid > 4000:
        sound_type = "high-centroid sound."
        sensitivity = "high"
        onset_delta = 0.02
        onset_backtrack = False
    else:
        sound_type = "mid-centroid sound."
        sensitivity = "mid"
        onset_delta = 0.05
        onset_backtrack = True
    
    print(f"  Detected {sound_type}")
    print(f"  Using {sensitivity}-sensitivity onset params.")
    
    # Onset detection with adjusted parameters
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', delta=onset_delta, backtrack=onset_backtrack)
    num_onsets = len(onset_times)
    print(f"  Detected onsets: {num_onsets}")
    
    if num_onsets < 2:
        print("  Insufficient onsets for rhythm analysis.")
        return
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Rhythm coherence: inverse of coefficient of variation
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 1 / (1 + cv)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice: adaptive base using GCD of rounded IOIs in ms
    iois_ms = np.round(iois * 1000).astype(int)
    if len(iois_ms) > 0 and np.all(iois_ms > 0):
        gcd_val = reduce(gcd, iois_ms)
        base = max(gcd_val / 1000.0, 0.001)  # Ensure minimum base
    else:
        base = 0.001
    print(f"  Rhythm lattice base: {base:.3f} s")
    
    # Lattice coherence: fraction of onsets fitting the lattice within tolerance
    if num_onsets > 0:
        rel_times = onset_times - onset_times[0]
        tol = base * 0.5
        fits = []
        for rt in rel_times[1:]:
            k = round(rt / base)
            dist = abs(rt - k * base)
            fits.append(dist < tol)
        lattice_coherence = np.mean(fits)
    else:
        lattice_coherence = 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Improved CQT: higher resolution for better shift invariance
    bins_per_octave = 48
    n_bins = 336  # 7 octaves * 48
    cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=n_bins, bins_per_octave=bins_per_octave)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # CQT pitch shift invariance metric (lower is better)
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    cqt_shift = librosa.cqt(y_shift, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    log_shift = librosa.amplitude_to_db(np.abs(cqt_shift))
    
    bps = bins_per_octave / 12  # bins per semitone
    shift_bins = int(bps)
    # Roll shifted CQT down to align
    rolled_shift = np.roll(log_shift, -shift_bins, axis=0)
    
    # Compute RMSE ignoring edge bins
    min_time = min(log_cqt.shape[1], rolled_shift.shape[1])
    diff = log_cqt[shift_bins:, :min_time] - rolled_shift[:-shift_bins, :min_time]
    metric = np.sqrt(np.mean(diff ** 2))
    print(f"  CQT shift invariance metric: {metric:.2f} (lower is more invariant)")

def main():
    print("Analyzing available WAV files.")
    available_files = [f for f in os.listdir('.') if f.endswith('.wav')]
    if not available_files:
        # Fallback to synthetic test signals
        print("No WAV files found. Generating synthetic test signals.")
        sr = 22050
        t = np.linspace(0, 5, 5 * sr)
        y_sine = np.sin(2 * np.pi * 440 * t)  # Simple sine wave
        librosa.output.write_wav('synthetic_sine.wav', y_sine, sr)
        analyze_audio('synthetic_sine.wav')
        os.remove('synthetic_sine.wav')
    else:
        for file in ['birdsong.wav', 'orchestra.wav', 'rock.wav']:
            if file in available_files:
                analyze_audio(file)

if __name__ == "__main__":
    main()