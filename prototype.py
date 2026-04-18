# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Bugfix, Rhythm Lattice Improvements, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def generate_synthetic_signal():
    sr = 22050
    duration = 10
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = [220, 440, 660, 880]  # Broader frequency content for testing invariance
    y = sum(np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi) for f in freqs)  # Phase randomization for coherence
    # Add rhythmic modulation with subdivisions for lattice testing
    beat_rate = 2.0  # ~120 BPM
    sub_rate = beat_rate * 4  # Quarter-note subdivisions
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * beat_rate * t) * (1 + 0.5 * np.sin(2 * np.pi * sub_rate * t))
    y *= envelope
    # Add noise for broad sound handling
    noise = np.random.normal(0, 0.1, len(t))
    y += noise
    y = librosa.util.normalize(y)  # Normalize for coherence
    return y, sr

def process_audio(y, sr):
    hop_length = 512
    # CQT with enhancements for invariance (e.g., higher resolution bins)
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=96, bins_per_octave=24, filter_scale=1.5)
    print("CQT shape:", cqt.shape)
    
    # Onset strength with aggregate for better rhythm detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)
    
    # Updated tempo estimation for rhythm lattice
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop_length, aggregate=np.median)
    print("Estimated tempo:", tempo)
    
    # Beat tracking with tightness for coherence
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='frames', tightness=200)
    
    # Sync CQT to beats for tempo invariance and rhythm lattice
    cqt_mag = np.abs(cqt)
    cqt_sync = librosa.util.sync(cqt_mag, beats, aggregate=np.mean, pad=True, axis=1)
    print("Synced CQT shape:", cqt_sync.shape)
    
    # Improvements: Normalize for amplitude invariance
    cqt_sync = librosa.util.normalize(cqt_sync, axis=0)
    
    # For CQT shift invariance, convert to chroma-like (fold octaves)
    chroma_sync = librosa.feature.chroma_cqt(C=cqt_sync, bins_per_octave=24, n_octaves=8)
    print("Chroma synced shape:", chroma_sync.shape)
    
    # Rhythm lattice: Create subdivision grid (e.g., 4x subdivisions per beat)
    subdivisions = []
    for i in range(len(beats) - 1):
        beat_start = beats[i]
        beat_end = beats[i+1]
        sub_frames = np.linspace(beat_start, beat_end, 5, dtype=int)[:-1]  # 4 subs
        subdivisions.extend(sub_frames)
    subdivisions = np.unique(np.concatenate([beats, subdivisions]))
    # Resync to finer lattice for better rhythm representation
    lattice_sync = librosa.util.sync(cqt_mag, subdivisions, aggregate=np.mean, pad=True, axis=1)
    print("Rhythm lattice synced shape:", lattice_sync.shape)
    
    # Coherence: Compute self-similarity matrix on lattice for structural invariance
    ssm = librosa.segment.recurrence_matrix(lattice_sync, mode='affinity', sym=True)
    print("Self-similarity matrix shape:", ssm.shape)
    
    # Broad sound handling: If low tonal content, fallback to MFCC
    if np.mean(onset_env) < 0.1:  # Heuristic for non-rhythmic/noisy sounds
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=20)
        mfcc_sync = librosa.util.sync(mfcc, beats, aggregate=np.mean, pad=True, axis=1)
        print("Fallback MFCC synced shape:", mfcc_sync.shape)

# Main execution
wav_files = []  # From instructions: (none)
if not wav_files:
    print("No WAV files available, using synthetic test signal.")
    y, sr = generate_synthetic_signal()
    process_audio(y, sr)
else:
    for file in wav_files:
        y, sr = librosa.load(file, sr=22050)
        process_audio(y, sr)