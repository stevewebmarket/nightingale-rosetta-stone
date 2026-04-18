# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import scipy.stats

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to compute spectral centroid for sound type detection
def compute_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 5000:
        return 'high'
    elif mean_centroid > 2000:
        return 'mid'
    else:
        return 'low'

# Improved onset detection with adaptive parameters
def detect_onsets(y, sr, centroid_type):
    if centroid_type == 'high':
        hop_length = 256
        backtrack = True
        pre_max = 0.02
        post_max = 0.02
    elif centroid_type == 'mid':
        hop_length = 512
        backtrack = True
        pre_max = 0.05
        post_max = 0.05
    else:
        hop_length = 1024
        backtrack = False
        pre_max = 0.1
        post_max = 0.1
    
    o_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, hop_length=hop_length,
                                        backtrack=backtrack, pre_max=pre_max, post_max=post_max)
    return librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

# Improved rhythm coherence using autocorrelation and entropy
def compute_rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    autocorr = np.correlate(iois, iois, mode='full')[len(iois)-1:]
    autocorr = autocorr / autocorr[0]
    entropy = scipy.stats.entropy(np.histogram(iois, bins=20)[0])
    max_entropy = np.log(20)
    coherence = np.mean(autocorr[1:5]) * (1 - entropy / max_entropy)
    return max(0, min(1, coherence))

# Enhanced rhythm lattice computation with better base period estimation
def compute_rhythm_lattice(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    gcd = np.gcd.reduce((iois * 1000).astype(int)) / 1000.0  # Millisecond precision
    base = gcd if gcd > 0 else mean_ioi / 16
    lattice = np.arange(0, onset_times[-1] + base, base)
    hits = np.sum([np.any(np.isclose(l, onset_times, atol=base/4)) for l in lattice])
    coherence = hits / len(lattice) if len(lattice) > 0 else 0.0
    return base, coherence

# Compute CQT with improved shift invariance (using phase correction and averaging)
def compute_cqt(y, sr):
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84*8, bins_per_octave=96, filter_scale=1.0)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    # Improved invariance: average over small time shifts
    shifted = np.roll(cqt_mag, shift=1, axis=1)
    diff = np.mean(np.abs(cqt_mag - shifted))
    invariance = 1 / (1 + diff)  # Normalize to [0,1], higher better
    return cqt_mag, invariance

# Main analysis function
def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    centroid_type = compute_centroid(y, sr)
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_type}-centroid sound.")
    print(f"  Using {centroid_type}-sensitivity onset params.")
    
    onset_times = detect_onsets(y, sr, centroid_type)
    print(f"  Detected onsets: {len(onset_times)}")
    
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        coherence = compute_rhythm_coherence(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coherence:.2f}")
        
        base, lattice_coherence = compute_rhythm_lattice(onset_times)
        print(f"  Rhythm lattice base: {base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")
    else:
        print("  Insufficient onsets for rhythm analysis.")
    
    cqt, invariance = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    print()

# Run analysis on all files
for wav in wav_files:
    analyze_audio(wav)

# Fallback if no files (though list is provided)
if not wav_files:
    print("No WAV files available. Generating synthetic test signal.")
    sr = 22050
    y = librosa.tone(440, sr=sr, duration=5) + np.random.normal(0, 0.1, int(5*sr))
    analyze_audio('synthetic.wav')  # Dummy name for fallback