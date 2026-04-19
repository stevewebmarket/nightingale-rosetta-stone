# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import scipy

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files, fall back to synthetic signals
if not wav_files:
    # Synthetic test signals
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Synthetic birdsong-like: high freq chirps
    birdsong_synth = np.sum([np.sin(2 * np.pi * freq * t) * (np.random.rand(len(t)) > 0.99) for freq in [4000, 5000, 6000]], axis=0)
    # Synthetic orchestra-like: mid freq harmonics
    orchestra_synth = np.sum([np.sin(2 * np.pi * freq * t) for freq in [440, 880, 1320]], axis=0) + 0.5 * np.random.randn(len(t))
    # Synthetic rock-like: beat with mid freq
    rock_synth = np.sin(2 * np.pi * 440 * t) * scipy.signal.square(2 * np.pi * 2 * t) + 0.3 * np.random.randn(len(t))
    
    signals = {'synth_birdsong': birdsong_synth, 'synth_orchestra': orchestra_synth, 'synth_rock': rock_synth}
else:
    signals = {}
    sr = 22050
    for file in wav_files:
        y, _ = librosa.load(file, sr=sr)
        signals[file] = y

def detect_centroid_type(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid sound"
    elif mean_centroid < 1000:
        return "low-centroid sound"
    else:
        return "mid-centroid sound"

def compute_rhythm_metrics(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    if len(onsets) < 2:
        return 0, 0.0, 0.0
    
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    
    # Improved coherence: coefficient of variation (std / mean), lower is more coherent, invert for [0,1] scale
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    rhythm_coherence = 1 / (1 + cv)  # Now higher for more regular rhythms
    
    return len(onsets), mean_ioi, rhythm_coherence

def compute_rhythm_lattice(y, sr, mean_ioi):
    # Improved lattice: use autocorrelation to find dominant period
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    autocorr = scipy.signal.correlate(onset_env, onset_env, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = scipy.signal.find_peaks(autocorr)[0]
    if len(peaks) > 1:
        base_period = peaks[1] / sr  # First peak after zero
    else:
        base_period = mean_ioi / 10 if mean_ioi > 0 else 0.1
    
    # Lattice coherence: fraction of onsets aligning to lattice multiples
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    lattice_points = np.arange(0, len(y)/sr, base_period)
    alignments = 0
    for onset in onsets:
        if np.min(np.abs(lattice_points - onset)) < base_period / 2:
            alignments += 1
    lattice_coherence = alignments / len(onsets) if len(onsets) > 0 else 0.0
    
    return base_period, lattice_coherence

def compute_cqt_invariance(y, sr):
    # CQT with parameters for broader handling
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=168, bins_per_octave=24, filter_scale=1.0)
    
    # Improved invariance metric: compare original CQT with pitch-shifted signal
    # Shift by one semitone (factor 2^(1/12))
    shift_factor = 2 ** (1/12)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
    cqt_shifted = librosa.cqt(y_shifted, sr=sr, hop_length=512, n_bins=168, bins_per_octave=24, filter_scale=1.0)
    
    # Normalize and compute correlation
    cqt_norm = cqt / (np.linalg.norm(cqt) + 1e-8)
    cqt_shifted_norm = cqt_shifted / (np.linalg.norm(cqt_shifted) + 1e-8)
    
    # Roll the shifted CQT by one bin to align
    cqt_shifted_rolled = np.roll(cqt_shifted_norm, shift=1, axis=0)
    
    # Metric: cosine similarity (higher closer to 1 means more invariant)
    similarity = np.dot(cqt_norm.flatten(), cqt_shifted_rolled.flatten())
    
    return cqt.shape, similarity

# Analyze each signal
for name, y in signals.items():
    print(f"Analysis for {name}:")
    centroid_type = detect_centroid_type(y, sr)
    print(f"  Detected {centroid_type}.")
    
    num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(y, sr)
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    
    base_period, lattice_coherence = compute_rhythm_lattice(y, sr, mean_ioi)
    print(f"  Rhythm lattice base: {base_period:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    cqt_shape, invariance_metric = compute_cqt_invariance(y, sr)
    print(f"  CQT shape: {cqt_shape}, n_bins: {cqt_shape[0]}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")