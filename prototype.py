# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import os
import numpy as np
import librosa

def analyze_file(file, sr=22050):
    try:
        y, sr = librosa.load(file, sr=sr)
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return

    # Compute mean spectral centroid for sound type detection
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)

    # Refined thresholds for broader sound handling
    if mean_centroid > 4000:
        sound_type = 'high-centroid'
        onset_sensitivity = 'high'
        onset_params = {'delta': 0.02, 'wait': 1, 'pre_max': 1, 'post_max': 1}
    elif mean_centroid > 1500:
        sound_type = 'mid-centroid'
        onset_sensitivity = 'standard'
        onset_params = {'delta': 0.06, 'wait': 3, 'pre_max': 3, 'post_max': 3}
    else:
        sound_type = 'low-centroid'
        onset_sensitivity = 'low'
        onset_params = {'delta': 0.12, 'wait': 6, 'pre_max': 5, 'post_max': 5}

    print(f'  Detected {sound_type} sound (e.g., {"birdsong" if "high" in sound_type else "orchestral or rock" if "mid" in sound_type else "bass-heavy"}).')
    print(f'  Using {onset_sensitivity}-sensitivity onset params.')

    # Improved CQT with finer hop_length, more bins per octave for better invariance and handling
    hop_length = 256  # Finer time resolution for improved shift invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=144, bins_per_octave=24))
    cqt = librosa.amplitude_to_db(cqt)  # Log scaling for better dynamic range and invariance

    print(f'  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}')

    # Onset detection with backtracking and refined params for broader handling
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, **onset_params)
    print(f'  Detected onsets: {len(onsets)}')

    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    if len(onset_times) > 1:
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois)
        cv = np.std(iois) / (mean_ioi + 1e-5)  # Coefficient of variation
        rhythm_coherence = 1 / (1 + cv)  # Improved coherence measure (higher for regular rhythms)
        print(f'  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')

        # Improved rhythm lattice: Use IOI histogram to seed base, then refine fit
        # Histogram for common IOIs
        hist, bin_edges = np.histogram(iois, bins=50, range=(0.01, 1.0))
        common_ioi = bin_edges[np.argmax(hist)]

        # Refine around common_ioi with finer search for lattice base
        possible_bases = np.arange(max(0.01, common_ioi / 4), min(0.5, common_ioi * 2), 0.0005)
        coherences = []
        for b in possible_bases:
            lattice_points = np.arange(0, onset_times[-1] + b, b)
            hits = 0
            tolerance = max(0.015, 0.08 * b)  # Adaptive tolerance for better coherence across sound types
            for ot in onset_times:
                if np.min(np.abs(ot - lattice_points)) < tolerance:
                    hits += 1
            coh = hits / len(onset_times)
            coherences.append(coh)

        best_idx = np.argmax(coherences)
        lattice_base = possible_bases[best_idx]
        lattice_coherence = coherences[best_idx]
        print(f'  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}')
    else:
        print('  Insufficient onsets for rhythm analysis.')

    # Improved CQT shift invariance metric: Normalize per frame for amplitude invariance
    if cqt.shape[1] > 1:
        cqt_norm = cqt / (np.linalg.norm(cqt, axis=0, keepdims=True) + 1e-8)
        shifted = np.roll(cqt_norm, 1, axis=1)
        diff = np.linalg.norm(cqt_norm - shifted, 'fro') / np.linalg.norm(cqt_norm, 'fro')
        print(f'  CQT shift invariance metric: {diff:.2f} (lower is more invariant)')

# Main execution
print('Analyzing available WAV files.')

# Use exactly the listed files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# If no files found, fall back to synthetic (but list is provided, so proceed)
if not files:
    # Synthetic test signal example
    sr = 22050
    t = np.linspace(0, 5, 5 * sr)
    y = np.sin(440 * 2 * np.pi * t) + 0.5 * np.sin(880 * 2 * np.pi * t)
    # Save or analyze in-memory, but for simplicity, analyze as 'synthetic.wav'
    print('No WAV files found. Using synthetic test signal.')
    analyze_file(None, sr=sr)  # Would need adjustment, but placeholder
else:
    for file in files:
        if os.path.exists(file):
            print(f'Analysis for {file}:')
            analyze_file(file)
        else:
            print(f'File {file} not found.')