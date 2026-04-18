# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
from scipy.stats import entropy
from scipy.signal import correlate

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def classify_sound_type(y, sr):
    """Classify sound based on spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 2000:
        return "low-centroid"
    elif mean_centroid > 4000:
        return "high-centroid"
    else:
        return "mid-centroid"

def detect_onsets(y, sr, sensitivity='mid'):
    """Detect onsets with adjustable sensitivity."""
    if sensitivity == 'low':
        hop_length = 1024
        backtrack = False
    elif sensitivity == 'high':
        hop_length = 256
        backtrack = True
    else:  # mid
        hop_length = 512
        backtrack = True
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length, backtrack=backtrack)
    return onsets

def compute_rhythm_metrics(onsets, sr):
    """Compute mean IOI and rhythm coherence."""
    times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(times)
    if len(iois) == 0:
        return 0, 0
    mean_ioi = np.mean(iois)
    # Improved coherence: inverse of normalized std + entropy of binned IOIs
    norm_std = np.std(iois) / mean_ioi
    hist, _ = np.histogram(iois, bins=20)
    ent = entropy(hist + 1e-10) / np.log(20)  # Normalized entropy
    coherence = 1 / (1 + norm_std + ent)
    return mean_ioi, coherence

def compute_rhythm_lattice(mean_ioi, iois, sr):
    """Compute adaptive rhythm lattice base and coherence."""
    # Adaptive base: fraction of min IOI, floored to ms resolution
    min_ioi = np.min(iois) if len(iois) > 0 else mean_ioi
    base = max(0.001, min_ioi / 20)  # At least 1ms, up to min_ioi/20
    base = np.round(base, 3)  # Round to ms
    
    # Lattice coherence: how well IOIs fit multiples of base
    multiples = iois / base
    errors = np.abs(multiples - np.round(multiples))
    lattice_coherence = 1 - np.mean(errors) / 0.5  # Normalized, higher better
    return base, lattice_coherence

def compute_cqt(y, sr):
    """Compute CQT with parameters for better invariance."""
    # Increased bins per octave for better resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.5)
    cqt_mag = np.abs(cqt)
    return cqt_mag

def cqt_shift_invariance(cqt_mag):
    """Compute improved shift invariance metric."""
    # Normalize CQT
    cqt_norm = cqt_mag / (np.max(cqt_mag) + 1e-10)
    
    # Compute autocorrelation in frequency dimension for each time frame
    autocorrs = [correlate(frame, frame, mode='full') for frame in cqt_norm.T]
    autocorrs = np.array(autocorrs)
    
    # Mean peak width as invariance proxy (narrower peaks = less invariant? Wait, invert)
    # Actually, higher invariance if autocorrelation is flat/high off zero
    mean_corr_off_zero = np.mean(np.abs(autocorrs[:, 1:])) / np.mean(np.abs(autocorrs[:, 0]))
    
    # Additional: entropy of frequency distribution per frame
    entropies = [entropy(frame + 1e-10) for frame in cqt_norm.T]
    mean_entropy = np.mean(entropies) / np.log(cqt_norm.shape[0])
    
    # Combined metric: higher correlation off-zero and higher entropy indicate broader invariance
    metric = 0.5 * mean_corr_off_zero + 0.5 * mean_entropy
    return metric

def analyze_file(filename):
    y, sr = librosa.load(filename, sr=22050)
    sound_type = classify_sound_type(y, sr)
    print(f"Analysis for {filename}:")
    print(f"  Detected {sound_type} sound.")
    
    if sound_type == 'low-centroid':
        sensitivity = 'low'
    elif sound_type == 'high-centroid':
        sensitivity = 'high'
    else:
        sensitivity = 'mid'
    print(f"  Using {sensitivity}-sensitivity onset params.")
    
    onsets = detect_onsets(y, sr, sensitivity)
    print(f"  Detected onsets: {len(onsets)}")
    
    mean_ioi, rhythm_coherence = compute_rhythm_metrics(onsets, sr)
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    iois = np.diff(librosa.frames_to_time(onsets, sr=sr))
    lattice_base, lattice_coherence = compute_rhythm_lattice(mean_ioi, iois, sr)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt_mag = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt_mag.shape}, n_bins: {cqt_mag.shape[0]}")
    
    invariance = cqt_shift_invariance(cqt_mag)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print()

if __name__ == "__main__":
    for file in wav_files:
        analyze_file(file)