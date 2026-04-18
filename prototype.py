# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Coherence + CQT Optimization
# =============================================================================

import librosa
import numpy as np
import scipy

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, use synthetic signals
if not wav_files:
    # Synthetic test signals
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    # Synthetic birdsong-like: chirps
    birdsong_synth = np.sum([np.sin(2 * np.pi * (2000 + 1000 * i) * t) for i in range(5)], axis=0)
    # Synthetic orchestra-like: harmonics
    orchestra_synth = np.sum([np.sin(2 * np.pi * (440 * (i+1)) * t) * (1/(i+1)) for i in range(10)], axis=0)
    # Synthetic rock-like: beat with distortion
    beat = np.sin(2 * np.pi * 2 * t) > 0.5
    rock_synth = scipy.signal.sawtooth(2 * np.pi * 200 * t) * beat
    signals = {'synth_birdsong': birdsong_synth, 'synth_orchestra': orchestra_synth, 'synth_rock': rock_synth}
else:
    signals = {}
    sr = 22050
    for file in wav_files:
        y, _ = librosa.load(file, sr=sr)
        signals[file] = y

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    return mean_centroid

def detect_onsets(y, sr, sensitivity='high'):
    if sensitivity == 'high':
        delta = 0.02
        wait = 1
    elif sensitivity == 'mid':
        delta = 0.07
        wait = 4
    else:
        delta = 0.1
        wait = 8
    o_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr, delta=delta, wait=wait, backtrack=True)
    return onsets

def compute_iois(onset_times):
    iois = np.diff(onset_times)
    return iois

def rhythm_coherence(iois):
    if len(iois) < 2:
        return 0.0
    hist, bin_edges = np.histogram(iois, bins=20)
    dominant = bin_edges[np.argmax(hist)]
    spread = np.std(iois) / dominant if dominant > 0 else 0
    coherence = 1 / (1 + spread)
    return coherence

def estimate_rhythm_lattice(iois, sr):
    # Improved: Use autocorrelation to find base period
    if len(iois) < 2:
        return 0.0, 0.0
    autocorr = scipy.signal.correlate(iois, iois, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = scipy.signal.find_peaks(autocorr)[0]
    if len(peaks) > 1:
        base = peaks[1] / sr  # First peak after zero
    else:
        base = np.median(iois)
    # Lattice coherence: fraction of onsets aligning to multiples of base
    alignment_errors = np.mod(iois.cumsum(), base)
    aligned = np.sum(alignment_errors < base * 0.05) / len(iois)  # 5% tolerance
    return base, aligned

def compute_cqt(y, sr):
    # Optimized CQT params for better shift invariance: more bins, smaller hop
    cqt = librosa.cqt(y, sr=sr, hop_length=256, n_bins=384, bins_per_octave=48, fmin=librosa.note_to_hz('C1'))
    cqt_mag = np.abs(cqt)
    # Normalize for invariance
    cqt_mag = librosa.util.normalize(cqt_mag, axis=0)
    return cqt_mag

def cqt_shift_invariance(cqt, max_shift=5):
    # Improved metric: average correlation over small shifts
    corrs = []
    for shift in range(1, max_shift + 1):
        shifted = np.roll(cqt, shift, axis=1)
        corr = np.mean([scipy.stats.pearsonr(cqt[:, i], shifted[:, i])[0] for i in range(cqt.shape[1]) if i < cqt.shape[1] - shift])
        corrs.append(corr)
    return np.mean(corrs)

# Analyze each signal
for name, y in signals.items():
    print(f"Analysis for {name}:")
    centroid = compute_spectral_centroid(y, sr)
    if centroid > 3000:
        print("  Detected high-centroid sound.")
        sensitivity = 'high'
    elif centroid > 1000:
        print("  Detected mid-centroid sound.")
        sensitivity = 'mid'
    else:
        print("  Detected low-centroid sound.")
        sensitivity = 'low'
    print(f"  Using {sensitivity}-sensitivity onset params.")
    
    onsets = detect_onsets(y, sr, sensitivity)
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = compute_iois(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence(iois):.2f}")
        lattice_base, lattice_coh = estimate_rhythm_lattice(onset_times, sr)  # Pass times for better calc
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coh:.2f}")
    else:
        print("  No IOIs detected.")
    
    cqt = compute_cqt(y, sr)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = cqt_shift_invariance(cqt)
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")