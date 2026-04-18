# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, and Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def detect_sound_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        return "high-centroid sound (e.g., birdsong)", True
    elif centroid > 1000:
        return "mid-centroid sound (e.g., orchestral or rock)", False
    else:
        return "low-centroid sound (e.g., bass-heavy)", False

def compute_rhythm_lattice(onset_times, min_base=0.005):
    if len(onset_times) < 2:
        return min_base, 0.0
    iois = np.diff(onset_times)
    base = np.gcd.reduce(np.round(iois / min_base).astype(int)) * min_base
    base = max(base, min_base)
    lattice = np.arange(0, onset_times[-1] + base, base)
    hits = np.sum(np.isclose(np.mod(onset_times, base), 0, atol=base/10))
    coherence = hits / len(onset_times)
    return base, coherence

def compute_cqt_invariance(cqt):
    shifted = np.roll(cqt, 1, axis=0)
    diff = np.mean(np.abs(cqt - shifted)) / np.mean(np.abs(cqt))
    return diff

def analyze_audio(file_path, sr=22050):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    y, sr = librosa.load(file_path, sr=sr)
    sound_type, is_high_centroid = detect_sound_type(y, sr)
    
    hop_length = 256 if is_high_centroid else 512
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
    
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        coherence = 1 - np.std(iois) / mean_ioi
    else:
        mean_ioi = 0
        coherence = 0
    
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=12)
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]
    
    lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times)
    invariance_metric = compute_cqt_invariance(np.abs(cqt))
    
    print(f"Analysis for {os.path.basename(file_path)}:")
    print(f"  Detected {sound_type}, using {'high' if is_high_centroid else 'standard'} onset params.")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  Detected onsets: {len(onsets)}, mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)")

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    print("Analyzing available WAV files.")
    for file in files:
        analyze_audio(file)

if __name__ == "__main__":
    main()