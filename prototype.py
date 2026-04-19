# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np

def detect_sound_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid < 1000:
        return "low-centroid sound"
    elif centroid > 3000:
        return "high-centroid sound"
    else:
        return "mid-centroid sound"

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    
    # Improved coherence: use autocorrelation for rhythm regularity
    autocorr = np.correlate(iois - np.mean(iois), iois - np.mean(iois), mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    coherence = np.max(autocorr) / (np.sum(autocorr) + 1e-8)
    
    # Improved lattice: use median IOI and refine with histogram peaks
    hist, bin_edges = np.histogram(iois, bins=50)
    base = bin_edges[np.argmax(hist)] / 2  # Halve the peak for base unit
    
    # Lattice coherence: fit onsets to lattice
    lattice = np.arange(0, onset_times[-1] + base, base)
    hits = np.sum([np.min(np.abs(l - onset_times)) < base / 4 for l in lattice])
    lattice_coherence = hits / (len(lattice) + len(onset_times)) / 2
    
    return len(onset_times), mean_ioi, coherence, base, lattice_coherence

def compute_cqt_invariance(cqt):
    # Improved invariance: normalize and compute shift correlation
    cqt_norm = librosa.util.normalize(np.abs(cqt), axis=0)
    shifts = []
    for i in range(1, min(10, cqt.shape[1] - 1)):
        corr = np.corrcoef(cqt_norm[:, :-i].flatten(), cqt_norm[:, i:].flatten())[0, 1]
        shifts.append(corr)
    return np.mean(shifts) if shifts else 0.0

def analyze_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    
    # Sound type
    sound_type = detect_sound_type(y, sr)
    
    # Onsets with improved detection for broad sounds
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, hop_length=256)
    onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True)
    
    # Rhythm metrics
    num_onsets, mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_metrics(onset_times)
    
    # CQT with parameters for better invariance and broad handling
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=384, bins_per_octave=48, filter_scale=1.5)
    cqt_shape = cqt.shape
    n_bins = cqt_shape[0]
    invariance_metric = compute_cqt_invariance(cqt)
    
    print(f"Analysis for {file_path}:")
    print(f"  Detected {sound_type}.")
    print(f"  Detected onsets: {num_onsets}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt_shape}, n_bins: {n_bins}")
    print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")
    print()

def main():
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)

if __name__ == "__main__":
    main()