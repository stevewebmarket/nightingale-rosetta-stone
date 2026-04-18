# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

def analyze_audio(file_path, sr=22050):
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    y, sr = librosa.load(file_path, sr=sr)
    
    # Compute spectral centroid for sound type classification
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:  # Threshold for high-centroid sounds like birdsong
        print(f"  Detected high-centroid sound (e.g., birdsong or treble-heavy).")
        print("  Using high-sensitivity onset params.")
        delta = 0.02  # More sensitive for high-frequency content
        pre_max = 0.01
        post_max = 0.01
    else:
        print(f"  Detected mid-centroid sound (e.g., orchestral or rock).")
        print("  Using standard-sensitivity onset params.")
        delta = 0.05  # Adjusted for better detection in mid-range
        pre_max = 0.03
        post_max = 0.03
    
    # Onset detection with improved parameters for broader sound handling
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, delta=delta, pre_max=pre_max, post_max=post_max)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    print(f"  Detected onsets: {len(onset_times)}")
    
    # Mean IOI with empty handling
    if len(onset_times) < 2:
        mean_ioi = 0.0
    else:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois) if len(iois) > 0 else 0.0
    print(f"  mean IOI: {mean_ioi:.2f} s", end="")
    
    # Rhythm coherence: inverse of normalized std of IOIs (improved for low onset cases)
    if len(onset_times) < 3:
        rhythm_coherence = 0.0
    else:
        iois = np.diff(onset_times)
        rhythm_coherence = 1.0 / (1.0 + np.std(iois) / (mean_ioi + 1e-6))
    print(f", rhythm coherence: {rhythm_coherence:.2f}")
    
    # Improved rhythm lattice using tempo estimation for better coherence
    if len(onset_times) < 2:
        lattice_base = 0.100  # Fallback for sparse onsets
        lattice_coherence = 0.5
    else:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        lattice_base = 60.0 / tempo[0] if tempo[0] > 0 else np.nan
        # Lattice coherence based on autocorrelation peak
        ac = librosa.autocorrelate(onset_env)
        ac = ac / (np.max(ac) + 1e-6)
        peaks = librosa.util.peak_pick(ac, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.1, wait=10)
        lattice_coherence = np.mean(ac[peaks]) if len(peaks) > 0 else 0.0
    print(f"  Rhythm lattice base: {lattice_base:.3f} s" if not np.isnan(lattice_base) else "  Rhythm lattice base: nan s", end="")
    print(f", lattice coherence: {lattice_coherence:.2f}")
    
    # CQT with improved invariance (using log-magnitude for better shift robustness)
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=144, bins_per_octave=24)
    log_cqt = librosa.amplitude_to_db(np.abs(cqt))
    print(f"  CQT shape: {log_cqt.shape}, n_bins: {log_cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: mean frame-to-frame cosine distance (lower better)
    invariance = 0.0
    for t in range(1, log_cqt.shape[1]):
        vec1 = log_cqt[:, t-1]
        vec2 = log_cqt[:, t]
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
        invariance += 1.0 - cos_sim  # Distance
    invariance /= (log_cqt.shape[1] - 1) if log_cqt.shape[1] > 1 else 1
    print(f"  CQT shift invariance metric: {invariance:.2f} (lower is more invariant)")

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example (sine wave)
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=5)
        # Analyze synthetic, but for now, skip details
        print("Synthetic analysis placeholder.")
        return
    
    print("Analyzing available WAV files.")
    for file in wav_files:
        print(f"Analysis for {file}:")
        analyze_audio(file)

if __name__ == "__main__":
    main()