# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Argument Fix + Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

def classify_sound_type(y, sr):
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        return 'high-centroid', 'rock/percussive', 'tight'
    elif centroid > 2000:
        return 'mid-centroid', 'orchestral', 'balanced'
    else:
        return 'low-centroid', 'ambient', 'loose'

def build_rhythm_lattice(beats, onsets, tempo, hop_length):
    # Improved rhythm lattice: create a grid of rhythmic events with coherence across multiples
    # Enhance coherence by aligning to tempo multiples and subdivisions
    beat_intervals = np.diff(beats)
    if len(beat_intervals) == 0:
        return np.array([])
    base_interval = np.mean(beat_intervals)
    lattice = np.arange(0, len(beats) * base_interval, base_interval / 4)  # Subdivide for finer lattice
    # Snap onsets to lattice for coherence
    snapped_onsets = np.array([np.argmin(np.abs(lattice - o)) for o in onsets])
    unique_snapped = np.unique(snapped_onsets)
    return lattice[unique_snapped]

def analyze_audio(y, sr, detection_type):
    # Fixed: Added detection_type parameter to handle varying onset detection strategies
    # Broad sound handling: Adjust parameters based on type
    if detection_type == 'tight':
        hop_length = 256
        pre_max = 0.01
        post_max = 0.01
    elif detection_type == 'balanced':
        hop_length = 512
        pre_max = 0.03
        post_max = 0.03
    else:
        hop_length = 1024
        pre_max = 0.05
        post_max = 0.05

    # Onset detection with adjusted parameters for type
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, pre_max=pre_max, post_max=post_max, units='samples')

    # Tempo and beat tracking for rhythm foundation
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    # Build improved rhythm lattice with coherence
    lattice = build_rhythm_lattice(beats, onsets, tempo, hop_length)
    
    # CQT for pitch features with invariance enhancements
    # Improve CQT invariance: Use log-amplitude and normalize for shift invariance
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=84, bins_per_octave=12))
    cqt_log = librosa.amplitude_to_db(cqt)
    # Normalize per frame for basic invariance to overall level
    cqt_norm = cqt_log - np.mean(cqt_log, axis=0, keepdims=True)
    # For octave invariance, could fold into chroma, but here we keep full CQT with normalization

    # Output some analysis results (expandable for mapping)
    print(f"  Onsets detected: {len(onsets)}")
    print(f"  Tempo: {tempo:.2f} BPM")
    print(f"  Rhythm lattice points: {len(lattice)}")
    print(f"  CQT shape (invariant-normalized): {cqt_norm.shape}")

# List of available files
files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

print("Analyzing available WAV files.")

for file in files:
    if not os.path.exists(file):
        print(f"File {file} not found, skipping.")
        continue
    print(f"Analysis for {file}:")
    y, sr = librosa.load(file, sr=22050)
    type_id, example, detection = classify_sound_type(y, sr)
    print(f"  Detected {type_id} sound (e.g., {example}), using {detection} onset detection.")
    try:
        analyze_audio(y, sr, detection)
    except Exception as e:
        print(f"  Error analyzing {file}: {str(e)}")
    print("---")

# Fallback if no files (though list is not empty)
if not files:
    print("No WAV files available, generating synthetic test signal.")
    duration = 5.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(440 * 2 * np.pi * t)  # Simple sine wave
    type_id, example, detection = classify_sound_type(y, sr)
    print(f"Detected {type_id} sound (e.g., {example}), using {detection} onset detection.")
    analyze_audio(y, sr, detection)