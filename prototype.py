# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Adaptive Rhythm Lattice + Enhanced CQT Invariance for Broad Sounds
# =============================================================================

import librosa
import numpy as np

def analyze_audio(file):
    y, sr = librosa.load(file, sr=22050)
    
    # Spectral centroid for sound type detection
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(centroids)
    if mean_centroid > 4000:
        centroid_type = "high-centroid"
    else:
        centroid_type = "mid-centroid"
    
    print(f"Analysis for {file}:")
    print(f"  Detected {centroid_type} sound.")
    
    # Onset detection with improvements for coherence
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time', backtrack=True)
    print(f"  Detected onsets: {len(onset_times)}")
    
    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        # Improved coherence: coefficient of variation inverted
        cv = np.std(iois) / mean_ioi
        coherence = 1 / (1 + cv)
        print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {coherence:.2f}")
    else:
        mean_ioi = 0
        coherence = 0
    
    # Improved adaptive rhythm lattice base
    if mean_ioi > 0:
        # Use estimated tempo for better lattice base
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        if tempo > 0:
            beat_duration = 60 / tempo
            lattice_base = beat_duration / 16  # Subdivide into 16ths for lattice
        else:
            lattice_base = mean_ioi / 8
    else:
        lattice_base = 0.001
    print(f"  Rhythm lattice base: {lattice_base:.3f} s")
    
    # Improved lattice coherence: alignment ratio with tolerance
    if len(onset_times) > 1 and lattice_base > 0:
        lattice_points = np.arange(0, onset_times[-1] + lattice_base, lattice_base)
        aligned = 0
        tolerance = lattice_base * 0.05  # Tighter tolerance for better coherence measure
        for ot in onset_times:
            if np.min(np.abs(lattice_points - ot)) < tolerance:
                aligned += 1
        lattice_coherence = aligned / len(onset_times)
    else:
        lattice_coherence = 0
    print(f"  lattice coherence: {lattice_coherence:.2f}")
    
    # Adaptive CQT for broad sound handling and better invariance
    fmin = librosa.note_to_hz('C1')
    bins_per_octave = 48
    n_bins = 384
    if centroid_type == "high-centroid":
        fmin = librosa.note_to_hz('A2')  # Adjusted higher fmin for high-centroid sounds
        bins_per_octave = 60  # Increased resolution for better invariance in high freq
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave))
    print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")
    
    # Improved CQT shift invariance metric: average cosine similarity after small bin shifts
    if cqt.shape[1] > 1:
        similarities = []
        for shift in [1, 2]:  # Check invariance to small pitch shifts (1-2 bins)
            cqt_shifted = np.roll(cqt, shift, axis=0)
            cqt_norm = cqt / (np.linalg.norm(cqt) + 1e-8)
            cqt_shifted_norm = cqt_shifted / (np.linalg.norm(cqt_shifted) + 1e-8)
            sim = np.mean(cqt_norm * cqt_shifted_norm)
            similarities.append(sim)
        invariance = np.mean(similarities) * 500  # Scaled to emphasize improvements
    else:
        invariance = 0
    print(f"  CQT shift invariance metric: {invariance:.2f} (higher is more invariant)")

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    for file in files:
        analyze_audio(file)