# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Optimization and CQT Pitch-Shift Invariance
# =============================================================================

import numpy as np
import librosa
from librosa.feature.rhythm import tempo
import os

def analyze_audio(y, sr, filename):
    print(f"Analysis for {filename}:")
    
    # Spectral centroid for sound type detection
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        print("  Detected high-centroid sound.")
    elif centroid > 1000:
        print("  Detected mid-centroid sound.")
    else:
        print("  Detected low-centroid sound.")
    
    # Onset detection
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    num_onsets = len(onset_times)
    print(f"  Detected onsets: {num_onsets}")
    
    if num_onsets >= 2:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv = std_ioi / mean_ioi if mean_ioi > 0 else 0
        rhythm_coherence = max(0, min(1, 1 - cv))
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
        
        # Improved rhythm lattice using GCD of rounded IOIs in ms
        iois_ms = (iois * 1000).round().astype(int)
        gcd = np.gcd.reduce(iois_ms) if len(iois_ms) > 1 else iois_ms[0]
        tau = gcd / 1000.0
        if tau == 0:
            tau = mean_ioi / 10
        print(f"  Rhythm lattice base: {tau:.3f} s")
        
        # Improved lattice coherence
        if tau > 0:
            remainders = onset_times % tau
            phase = np.median(remainders)
            adjusted_rem = (remainders - phase) % tau
            std_rem = np.std(adjusted_rem)
            uniform_std = tau / np.sqrt(12)
            lattice_coherence = max(0, min(1, 1 - (std_rem / uniform_std)))
            print(f"  lattice coherence: {lattice_coherence:.2f}")
        else:
            print("  lattice coherence: 0.00")
    else:
        print("  Insufficient onsets for rhythm analysis.")
    
    # CQT with specified parameters for better invariance
    fmin = librosa.note_to_hz('C1')
    bpo = 48
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=384, bins_per_octave=bpo)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric (pitch-shift by 1 semitone)
    y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    cqt_shift = librosa.cqt(y_shift, sr=sr, fmin=fmin, n_bins=384, bins_per_octave=bpo)
    bins_shift = int(bpo / 12)
    abs_cqt = np.abs(cqt)
    abs_cqt_shift = np.abs(cqt_shift)
    rolled = np.roll(abs_cqt_shift, -bins_shift, axis=0)
    min_frames = min(abs_cqt.shape[1], rolled.shape[1])
    corr = np.corrcoef(abs_cqt[:, :min_frames].flatten(), rolled[:, :min_frames].flatten())[0, 1]
    print(f"  CQT shift invariance metric: {corr:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic test signals if no files
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y_sine = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 tone
        y_chirp = librosa.chirp(fmin=220, fmax=880, sr=sr, length=len(y_sine))
        synthetic_data = [("synthetic_sine", y_sine), ("synthetic_chirp", y_chirp)]
        for name, y in synthetic_data:
            analyze_audio(y, sr, f"{name}.wav (synthetic)")
    else:
        for file in files:
            if os.path.exists(file):
                y, sr = librosa.load(file, sr=22050)
                analyze_audio(y, sr, file)
            else:
                print(f"File {file} not found.")