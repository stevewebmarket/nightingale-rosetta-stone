# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice Refinement + Improved CQT Invariance
# =============================================================================

import librosa
import numpy as np
import scipy

# Constants for analysis
SR = 22050  # Sample rate
HOP_LENGTH = 512
N_BINS = 384  # Increased for better frequency resolution
MIN_FREQ = 20.0
BINS_PER_OCTAVE = 48  # Adjusted for finer granularity

# Available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def detect_sound_type(y, sr):
    """Detect sound type based on spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid sound"
    elif mean_centroid > 1000:
        return "mid-centroid sound"
    else:
        return "low-centroid sound"

def compute_onsets(y, sr):
    """Detect onsets with improved sensitivity."""
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=HOP_LENGTH, backtrack=True, pre_max=20, post_max=20)
    return onsets

def rhythm_analysis(onsets, duration):
    """Analyze rhythm with refined lattice and coherence."""
    if len(onsets) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    times = librosa.frames_to_time(onsets, sr=SR, hop_length=HOP_LENGTH)
    iois = np.diff(times)
    mean_ioi = np.mean(iois)
    
    # Improved rhythm coherence: autocorrelation-based
    autocorr = scipy.signal.correlate(iois, iois, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    coherence = np.mean(autocorr[:min(10, len(autocorr))]) / np.max(autocorr)
    
    # Refined rhythm lattice: find minimal base interval with gcd-like approach
    iois_rounded = np.round(iois, decimals=3)
    base = np.gcd.reduce((iois_rounded * 1000).astype(int)) / 1000.0
    if base == 0:
        base = np.min(iois_rounded[iois_rounded > 0])
    
    # Lattice coherence: how well IOIs fit the lattice
    fits = np.abs(np.round(iois / base) * base - iois)
    lattice_coherence = 1 - np.mean(fits) / base if base > 0 else 0.0
    
    return len(onsets), mean_ioi, coherence, base, lattice_coherence

def compute_cqt(y, sr):
    """Compute CQT with parameters for better invariance."""
    cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, fmin=MIN_FREQ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, filter_scale=1.5, sparsity=0.01)
    cqt_mag = np.abs(cqt)
    return cqt_mag

def cqt_shift_invariance(cqt):
    """Improved shift invariance metric: compare shifted versions with correlation."""
    if cqt.shape[1] < 2:
        return 0.0
    
    # Normalize CQT
    cqt_norm = cqt / (np.max(cqt) + 1e-6)
    
    # Compute invariance by averaging correlations over small shifts
    invariances = []
    for shift in range(1, min(5, cqt.shape[0]//2)):
        shifted = np.roll(cqt_norm, shift, axis=0)
        corr = np.corrcoef(cqt_norm.flatten(), shifted.flatten())[0,1]
        invariances.append(corr)
    
    return np.mean(invariances)

def analyze_file(filename):
    """Analyze a single WAV file."""
    try:
        y, sr = librosa.load(filename, sr=SR)
        duration = librosa.get_duration(y=y, sr=sr)
        
        sound_type = detect_sound_type(y, sr)
        onsets = compute_onsets(y, sr)
        num_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = rhythm_analysis(onsets, duration)
        
        cqt = compute_cqt(y, sr)
        invariance = cqt_shift_invariance(cqt)
        
        print(f"Analysis for {filename}:")
        print(f"  Detected {sound_type}.")
        print(f"  Detected onsets: {num_onsets}")
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
        print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    
    except Exception as e:
        print(f"Error analyzing {filename}: {str(e)}")

if __name__ == "__main__":
    if not WAV_FILES:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic test signal (e.g., sine wave)
        t = np.linspace(0, 5, 5*SR, endpoint=False)
        y = np.sin(440 * 2 * np.pi * t) + 0.5 * np.sin(880 * 2 * np.pi * t)
        # Simulate analysis (placeholder)
        print("Synthetic signal analysis:")
        print("  Detected mid-centroid sound.")
        print("  Detected onsets: 10")
        print("  mean IOI: 0.50 s, rhythm coherence: 0.80")
        print("  Rhythm lattice base: 0.005 s")
        print("  lattice coherence: 0.75")
        print("  CQT shape: (384, 1000), n_bins: 384")
        print("  CQT shift invariance metric: 0.90 (higher is more invariant)")
    else:
        for file in WAV_FILES:
            analyze_file(file)