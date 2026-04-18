# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Lattice Dynamics and CQT Adaptivity
# =============================================================================

import librosa
import numpy as np
import math

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    
    # Compute spectral centroid to classify sound type
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    
    if mean_centroid > 5000:
        print("  Detected high-centroid sound.")
        print("  Using high-sensitivity onset params.")
        # High-sensitivity params for transient-rich sounds like birdsong
        onset_hop = 128
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=onset_hop, aggregate=np.median)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, pre_max=0.02, post_max=0.02)
        cqt_hop = 128  # Smaller hop for better temporal resolution in high-frequency content
        cqt_bpo = 48   # Standard bins per octave
    else:
        print("  Detected mid-centroid sound.")
        print("  Using mid-sensitivity onset params.")
        onset_hop = 512
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=onset_hop)
        cqt_hop = 512
        cqt_bpo = 48
    
    num_onsets = len(onsets)
    print(f"  Detected onsets: {num_onsets}")
    
    # Compute IOIs
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=onset_hop)
    iois = np.diff(onset_times)
    if len(iois) > 0:
        mean_ioi = np.mean(iois)
    else:
        mean_ioi = 0
    print(f"  mean IOI: {mean_ioi:.2f} s", end="")
    
    # Rhythm coherence: 1 - coefficient of variation (improved normalization)
    if len(iois) > 1:
        cv = np.std(iois) / mean_ioi
        coherence = max(0, 1 - cv) if cv <= 1 else 0
    else:
        coherence = 0
    print(f", rhythm coherence: {coherence:.2f}")
    
    # Improved rhythm lattice base: use GCD of quantized IOIs for better adaptability
    if len(iois) > 0:
        iois_ms = [int(round(i * 1000)) for i in iois if i > 0]
        if iois_ms:
            base_ms = math.gcd(*iois_ms)
            lattice_base = max(0.001, base_ms / 1000.0)
        else:
            lattice_base = 0.001
    else:
        lattice_base = 0.001
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    
    # Improved lattice coherence: proportion of onsets aligning within adaptive tolerance
    tolerance = lattice_base * 0.1  # Tighter tolerance for better coherence measurement
    aligned = 0
    ref_time = onset_times[0] if len(onset_times) > 0 else 0
    for t in onset_times:
        rel_t = t - ref_time
        multiples = round(rel_t / lattice_base)
        aligned_t = multiples * lattice_base + ref_time
        if abs(t - aligned_t) <= tolerance:
            aligned += 1
    lattice_coherence = aligned / num_onsets if num_onsets > 0 else 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with adaptive parameters for better invariance
    n_bins = 384
    cqt = librosa.cqt(y=y, sr=sr, hop_length=cqt_hop, n_bins=n_bins, bins_per_octave=cqt_bpo)
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: average correlation over multiple small shifts
    if cqt.shape[1] > 5:  # Ensure enough frames
        cqt_mag = np.abs(cqt)
        invariance_scores = []
        for shift in [1, 2]:  # Check invariance to small shifts
            shifted = np.roll(cqt_mag, shift, axis=1)
            corrs = [np.corrcoef(cqt_mag[i], shifted[i])[0, 1] for i in range(n_bins)]
            invariance_scores.append(np.mean(corrs))
        invariance = np.mean(invariance_scores)
    else:
        invariance = 0
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

# List of available WAV files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Analyze each file
for file in files:
    print(f"Analysis for {file}:")
    analyze_audio(file)
    print("")