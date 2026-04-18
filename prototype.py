# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance
# =============================================================================

import librosa
import numpy as np
import os
from scipy.stats import entropy
from math import gcd
from functools import reduce

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1500:
        return 'low-centroid'
    elif mean_centroid < 4000:
        return 'mid-centroid'
    else:
        return 'high-centroid'

def get_onset_params(sound_type):
    if sound_type == 'low-centroid':
        return {'pre_max': 0.03, 'post_max': 0.03, 'pre_avg': 0.03, 'post_avg': 0.03, 'wait': 0.03, 'delta': 0.1, 'backtrack': True}
    elif sound_type == 'mid-centroid':
        return {'pre_max': 0.02, 'post_max': 0.02, 'pre_avg': 0.02, 'post_avg': 0.02, 'wait': 0.02, 'delta': 0.07, 'backtrack': True}
    else:
        return {'pre_max': 0.01, 'post_max': 0.01, 'pre_avg': 0.01, 'post_avg': 0.01, 'wait': 0.01, 'delta': 0.05, 'backtrack': True}

def compute_rhythm_coherence(iois, tempo):
    if len(iois) < 2:
        return 0.0
    std_ioi = np.std(iois)
    mean_ioi = np.mean(iois)
    cv = std_ioi / mean_ioi
    tempo_consistency = 1 - (std_ioi / (60 / tempo))  # Normalize against estimated beat interval
    return max(0, min(1, 1 - cv * (1 - tempo_consistency)))

def find_rhythm_lattice_base(iois):
    if len(iois) < 2:
        return 0.001
    iois_ms = [int(ioi * 1000) for ioi in iois if ioi > 0]
    if not iois_ms:
        return 0.001
    gcd_ms = reduce(gcd, iois_ms)
    return max(0.001, gcd_ms / 1000.0)

def compute_lattice_coherence(onset_times, lattice_base):
    if len(onset_times) < 2:
        return 0.0
    quantized = np.round(onset_times / lattice_base) * lattice_base
    errors = np.abs(onset_times - quantized)
    return 1 - np.mean(errors) / lattice_base

def compute_cqt_invariance(cqt, hop_length, sr):
    if cqt.shape[1] < 2:
        return 0.0
    # Normalize CQT for better invariance
    cqt_norm = librosa.util.normalize(np.log1p(np.abs(cqt)), axis=0)
    # Shift by one frame
    shifted = np.roll(cqt_norm, 1, axis=1)
    # Compute cosine similarity across time
    sim = np.mean([np.dot(cqt_norm[:, i], shifted[:, i]) / (np.linalg.norm(cqt_norm[:, i]) * np.linalg.norm(shifted[:, i]) + 1e-8) for i in range(cqt.shape[1])])
    # Metric: 1 - similarity (lower means more invariant? Wait, no: higher similarity means more invariant to shift)
    # Adjust: lower metric means more invariant, so metric = 1 - sim
    return 1 - sim

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    sound_type = compute_spectral_centroid(y, sr)
    print(f"  Detected {sound_type} sound.")
    params = get_onset_params(sound_type)
    print(f"  Using {sound_type.split('-')[0]}-sensitivity onset params.")
    
    # Improved onset detection with tempo hint
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='frames', hop_length=512, backtrack=params['backtrack'], **{k: v for k, v in params.items() if k != 'backtrack'})
    print(f"  Detected onsets: {len(onsets)}")
    
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois) if len(iois) > 0 else 0
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {compute_rhythm_coherence(iois, tempo):.2f}")
    
    lattice_base = find_rhythm_lattice_base(iois)
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {compute_lattice_coherence(onset_times, lattice_base):.2f}")
    
    # Improved CQT with more bins per octave for better frequency resolution and invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=420, bins_per_octave=60, fmin=librosa.note_to_hz('C1'))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    invariance = compute_cqt_invariance(cqt, 512, sr)
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    print("Analyzing available WAV files.")
    for file in wav_files:
        if os.path.exists(file):
            print(f"Analysis for {file}:")
            analyze_audio(file)
        else:
            print(f"File {file} not found, skipping.")

    if not wav_files:
        print("No WAV files available, using synthetic test signal.")
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=5) + 0.5 * librosa.tone(660, sr=sr, duration=5)
        # Save temporarily or analyze directly
        sound_type = compute_spectral_centroid(y, sr)
        print(f"  Detected {sound_type} sound for synthetic signal.")
        # Proceed similarly...

if __name__ == "__main__":
    main()