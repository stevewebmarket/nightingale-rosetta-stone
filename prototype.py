# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Lattice + Enhanced CQT Invariance
# =============================================================================

import librosa
import numpy as np
import scipy

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid > 3000:
        return "high-centroid"
    elif mean_centroid > 1000:
        return "mid-centroid"
    else:
        return "low-centroid"

def compute_rhythm_metrics(onset_times):
    if len(onset_times) < 2:
        return 0, 0.0, 0.0
    iois = np.diff(onset_times)
    mean_ioi = np.mean(iois)
    # Improved coherence: use coefficient of variation (lower is more coherent)
    cv = np.std(iois) / mean_ioi if mean_ioi > 0 else 0
    coherence = 1 / (1 + cv)  # Normalize to [0,1], higher better
    return len(onset_times), mean_ioi, coherence

def compute_rhythm_lattice(onset_times, mean_ioi):
    if len(onset_times) < 2:
        return 0.001, 0.0
    # Adaptive lattice base: approx 1/100th of mean IOI for finer grid
    base = mean_ioi / 100
    # Quantize onsets to lattice
    quantized = np.round(onset_times / base) * base
    # Lattice coherence: mean absolute error normalized
    errors = np.abs(onset_times - quantized)
    max_error = base / 2
    lattice_coherence = 1 - np.mean(errors) / max_error
    return base, lattice_coherence

def compute_cqt_invariance(cqt):
    # Improved shift invariance: compute autocorrelation along time axis for each bin
    # Then average normalized autocorrelation peaks
    invariance_scores = []
    for bin_idx in range(cqt.shape[0]):
        bin_series = np.abs(cqt[bin_idx, :])
        if np.max(bin_series) == 0:
            continue
        bin_series /= np.max(bin_series)  # Normalize
        autocorr = scipy.signal.correlate(bin_series, bin_series, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Half
        autocorr /= autocorr[0] if autocorr[0] != 0 else 1  # Normalize
        # Find average of top peaks (excluding zero lag)
        peaks = scipy.signal.find_peaks(autocorr, height=0.1)[0]
        if len(peaks) > 1:  # At least one shift
            invariance_scores.append(np.mean(autocorr[peaks[1:]]))  # Mean of non-zero lag peaks
    if not invariance_scores:
        return 0.0
    return np.mean(invariance_scores)

def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        centroid_type = compute_spectral_centroid(y, sr)
        print(f"Analysis for {file_path}:")
        print(f"  Detected {centroid_type} sound.")

        # Onset detection with adaptive backtracking for broad sound handling
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
        num_onsets, mean_ioi, rhythm_coherence = compute_rhythm_metrics(onsets)
        print(f"  Detected onsets: {num_onsets}")
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")

        # Adaptive rhythm lattice
        lattice_base, lattice_coherence = compute_rhythm_lattice(onsets, mean_ioi)
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")
        print(f"  lattice coherence: {lattice_coherence:.2f}")

        # CQT with parameters tuned for invariance and broad handling
        # Use higher resolution for high-centroid sounds
        hop_length = 256 if 'high' in centroid_type else 512
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=384, bins_per_octave=48)
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

        # Improved invariance metric
        invariance = compute_cqt_invariance(cqt)
        print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

if __name__ == "__main__":
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example (sine wave)
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        # Wrap in a dummy file_path for analysis
        analyze_audio('synthetic.wav')  # But actually pass y directly; modify if needed
    else:
        for wav in wav_files:
            analyze_audio(wav)