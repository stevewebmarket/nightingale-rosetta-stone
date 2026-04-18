# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import scipy.signal
import scipy.stats

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def classify_centroid(mean_centroid):
    if mean_centroid < 1500:
        return "low-centroid"
    elif mean_centroid < 5000:
        return "mid-centroid"
    else:
        return "high-centroid"

def compute_rhythm_lattice(onsets, sr, centroid_class):
    if len(onsets) < 2:
        return 0.0, 0.0
    
    iois = np.diff(onsets)
    mean_ioi = np.mean(iois)
    std_ioi = np.std(iois)
    rhythm_coherence = 1 / (1 + std_ioi / (mean_ioi + 1e-6))  # Normalized coherence
    
    # Improved rhythm lattice: Use autocorrelation for base period estimation
    onset_times = onsets
    max_lag = min(100, len(onset_times) - 1)  # Limit lags for efficiency
    autocorr = np.correlate(iois, iois, mode='full')[len(iois)-1:]
    autocorr = autocorr[:max_lag]
    peaks = scipy.signal.find_peaks(autocorr)[0]
    
    if len(peaks) > 0:
        base_lag = peaks[0] + 1  # First peak lag
        lattice_base = mean_ioi * base_lag
    else:
        lattice_base = mean_ioi * 2  # Fallback to double mean
    
    # Adjust based on centroid for broad handling
    if centroid_class == "low-centroid":
        lattice_base *= 1.5  # Emphasize longer periods for bass-heavy
    elif centroid_class == "high-centroid":
        lattice_base *= 0.8  # Shorter for high freq
    
    # Lattice coherence: Fit onsets to lattice grid
    grid = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
    hits = 0
    for ot in onset_times:
        if np.min(np.abs(grid - ot)) < lattice_base * 0.1:  # Tolerance 10%
            hits += 1
    lattice_coherence = hits / len(onset_times) if len(onset_times) > 0 else 0.0
    
    return mean_ioi, rhythm_coherence, lattice_base, lattice_coherence

def compute_cqt_invariance(y, sr, n_bins=384):
    # Compute HCQT for improved octave invariance
    hop_length = 256  # Smaller hop for better resolution
    fmin = librosa.note_to_hz('C1')
    hcqt = librosa.core.hybrid_cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=48, tuning=0.0)
    
    # For shift invariance metric: Compute self-similarity after synthetic octave shift
    # Synthetic shift: Resample y to simulate octave up (double speed)
    y_shifted = scipy.signal.resample(y, int(len(y) * 2))
    hcqt_shifted = librosa.core.hybrid_cqt(y_shifted, sr=sr*2, hop_length=hop_length, fmin=fmin*2, n_bins=n_bins, bins_per_octave=48, tuning=0.0)
    
    # Normalize magnitudes
    hcqt_mag = np.abs(hcqt)
    hcqt_shifted_mag = np.abs(hcqt_shifted)
    hcqt_mag /= np.max(hcqt_mag) + 1e-6
    hcqt_shifted_mag /= np.max(hcqt_shifted_mag) + 1e-6
    
    # Roll shifted to align octaves (shift by 48 bins for one octave with 48 bpo)
    hcqt_shifted_rolled = np.roll(hcqt_shifted_mag, -48, axis=0)
    
    # Compute correlation as invariance metric
    min_len = min(hcqt_mag.shape[1], hcqt_shifted_rolled.shape[1])
    corr = scipy.stats.pearsonr(hcqt_mag[:, :min_len].flatten(), hcqt_shifted_rolled[:, :min_len].flatten())[0]
    invariance = max(0, corr)  # Clamp to [0,1]
    
    return hcqt, invariance

def analyze_audio(filename):
    y, sr = librosa.load(filename, sr=22050)
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    centroid_class = classify_centroid(mean_centroid)
    
    # Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, backtrack=True)  # Improved with backtrack
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    # Rhythm lattice
    mean_ioi, rhythm_coherence, lattice_base, lattice_coherence = compute_rhythm_lattice(onset_times, sr, centroid_class)
    
    # CQT with improved invariance
    cqt, invariance = compute_cqt_invariance(y, sr)
    
    print(f"Analysis for {filename}:")
    print(f"  Detected {centroid_class} sound.")
    print(f"  Detected onsets: {len(onset_times)}")
    print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    for wav in wav_files:
        analyze_audio(wav)
        print()  # Blank line between analyses