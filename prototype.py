# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Coherence + CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=22050)
    
    # Compute spectral centroid for sound type classification
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 4000:
        sound_type = "high-centroid sound (e.g., birdsong)"
        sensitivity = "high-sensitivity"
        delta = 0.04  # Increased sensitivity for high-frequency content
        backtrack = False
    elif mean_centroid < 1000:
        sound_type = "low-centroid sound (e.g., bass-heavy)"
        sensitivity = "low-sensitivity"
        delta = 0.10
        backtrack = True
    else:
        sound_type = "mid-centroid sound (e.g., orchestral or rock)"
        sensitivity = "standard-sensitivity"
        delta = 0.07
        backtrack = True
    
    print(f"  Detected {sound_type}.")
    print(f"  Using {sensitivity} onset params.")
    
    # Onset detection with adjusted parameters
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=backtrack, delta=delta)
    print(f"  Detected onsets: {len(onsets)}")
    
    # Compute IOIs and rhythm coherence
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    rhythm_coherence = 1 - (np.std(iois) / mean_ioi) if mean_ioi > 0 and len(iois) > 1 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice: Use tempogram to find dominant periods, select best for coherence
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    autocorr = np.mean(tempogram, axis=0)  # Average over time for periodicity
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
    candidate_bases = [60 / (i + 1) for i in peaks if i > 0]  # Convert lags to periods (approx)
    if not candidate_bases:
        candidate_bases = [mean_ioi]
    
    best_coherence = 0
    best_base = candidate_bases[0]
    for base in candidate_bases + [mean_ioi]:  # Include mean IOI as fallback
        if base <= 0:
            continue
        quantized = np.round(times / base) * base
        errors = np.abs(times - quantized) / base
        coherence = 1 - np.mean(errors) if len(errors) > 0 else 0
        if coherence > best_coherence:
            best_coherence = coherence
            best_base = base
    
    print(f"  Rhythm lattice base: {best_base:.3f} s, lattice coherence: {best_coherence:.2f}")
    
    # CQT with improved invariance: Use higher bins_per_octave for better shift invariance
    cqt = librosa.cqt(y, sr=sr, n_bins=144, bins_per_octave=36, fmin=librosa.note_to_hz('C1'))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved shift invariance metric: Normalized difference after log-mag and shift
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    shifted = np.roll(log_cqt, 1, axis=0)
    diff = np.mean(np.abs(log_cqt - shifted))
    max_val = np.max(log_cqt) - np.min(log_cqt) if np.ptp(log_cqt) > 0 else 1
    invariance = diff / max_val
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

# Main execution
wav_files = [f for f in os.listdir('.') if f.lower().endswith('.wav')]
if not wav_files:
    print("No WAV files found. Falling back to synthetic test signals.")
    # Synthetic signal example (sine wave with rhythm)
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.1)  # Pulsed tone
    # Save or directly analyze synthetic
    analyze_file('synthetic.wav')  # Placeholder, but in code we'd process y directly
else:
    print("Analyzing available WAV files.")
    for file in sorted(wav_files):
        print(f"Analysis for {file}:")
        analyze_file(file)