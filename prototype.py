# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np

def classify_sound_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def compute_rhythm_metrics(y, sr, sound_type):
    if sound_type == "high-centroid":
        hop_length = 256  # Finer resolution for high-frequency content
    else:
        hop_length = 512
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    
    if len(onset_times) < 2:
        return len(onset_times), 0.0, 0.0, 0.001, 0.0
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved coherence: use coefficient of variation, inverted for coherence (lower CV = higher coherence)
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 1 / (1 + cv)
    
    # Adaptive rhythm lattice base: based on gcd of quantized IOIs
    quantized_iois = np.round(iois * 1000).astype(int)  # to ms
    gcd = np.gcd.reduce(quantized_iois) / 1000.0 if len(quantized_iois) > 0 else 0.001
    lattice_base = max(gcd, 0.001)  # Ensure minimum 1ms
    
    # Lattice coherence: fraction of onsets fitting multiples of lattice base
    fits = np.sum(np.isclose(np.mod(onset_times[1:], lattice_base), 0, atol=0.01)) / (len(onset_times) - 1)
    lattice_coherence = fits
    
    return len(onset_times), mean_ioi, rhythm_coherence, lattice_base, lattice_coherence

def compute_cqt_metrics(y, sr, sound_type):
    if sound_type == "high-centroid":
        fmin = librosa.note_to_hz('C4')  # Higher fmin for birdsong
        n_bins = 48  # More bins for broader frequency handling
        bins_per_octave = 24  # Increased for better resolution
    else:
        fmin = librosa.note_to_hz('C1')
        n_bins = 84
        bins_per_octave = 12
    
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
    
    # Improved shift invariance metric: average correlation across octave shifts
    invariance = 0.0
    num_shifts = min(3, n_bins // bins_per_octave)  # Up to 3 octaves
    for shift in range(1, num_shifts + 1):
        shifted = np.roll(cqt, shift * bins_per_octave, axis=0)
        corr = np.corrcoef(cqt.flatten(), shifted.flatten())[0, 1]
        invariance += corr
    invariance /= num_shifts if num_shifts > 0 else 1
    invariance = max(0.0, min(1.0, invariance))  # Clamp to [0,1]
    
    return cqt.shape, n_bins, invariance

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=22050)
    sound_type = classify_sound_type(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type} sound.")
    
    num_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(y, sr, sound_type)
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt_shape, n_bins, invariance = compute_cqt_metrics(y, sr, sound_type)
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ["birdsong.wav", "orchestra.wav", "rock.wav"]
    for file in files:
        analyze_file(file)