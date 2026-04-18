# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Index Bounds + Enhanced Rhythm Lattice & CQT
# =============================================================================

import librosa
import numpy as np
import os

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def analyze_audio(file, y, sr):
    mean_centroid = compute_spectral_centroid(y, sr)
    
    if mean_centroid > 5000:
        print(f"  Detected high-centroid sound (e.g., birdsong), using strict onset detection.")
        hop_length = 256
        pre_max = 0.01
        post_max = 0.01
        delta = 0.1
    elif mean_centroid > 1500:
        print(f"  Detected mid-centroid sound (e.g., orchestral or rock), using balanced onset detection.")
        hop_length = 512
        pre_max = 0.03
        post_max = 0.03
        delta = 0.07
    else:
        print(f"  Detected low-centroid sound (e.g., bass-heavy), using relaxed onset detection.")
        hop_length = 1024
        pre_max = 0.05
        post_max = 0.05
        delta = 0.05
    
    # Compute CQT with adaptive hop_length for broad sound handling
    n_bins = 84  # Base bins, adjustable for invariance
    fmin = 32.7 if mean_centroid < 1500 else 65.4  # Adjust fmin for low vs mid/high
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins * 2, fmin=fmin))
    
    # For CQT invariance: apply log-frequency scaling and normalization
    S = librosa.amplitude_to_db(cqt)
    S_norm = (S - np.min(S)) / (np.max(S) - np.min(S))  # Normalize for invariance
    
    # Onset detection on the CQT to ensure matching dimensions and coherence
    novelty = librosa.onset.onset_strength(sr=sr, S=S_norm, aggregate=np.median)
    
    # Improve coherence with smoothing
    novelty = librosa.util.normalize(novelty)
    novelty = np.convolve(novelty, np.ones(5)/5, mode='same')  # Simple smoothing for coherence
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=novelty,
        sr=sr,
        hop_length=hop_length,
        pre_max=pre_max,
        post_max=post_max,
        delta=delta,
        backtrack=True  # Enable backtrack for better rhythm accuracy
    )
    
    # Build improved rhythm lattice: compute temporal differences and cluster for lattice structure
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    deltas = np.diff(onset_times)
    deltas = deltas[deltas > 0]  # Filter invalid
    
    # Lattice: quantize deltas to a grid for rhythm lattice (improved coherence)
    if len(deltas) > 0:
        tempo, _ = librosa.beat.beat_track(onset_envelope=novelty, sr=sr, hop_length=hop_length)
        beat_duration = 60 / tempo
        lattice_grid = np.arange(0.25 * beat_duration, max(deltas) + beat_duration, 0.25 * beat_duration)
        quantized_deltas = np.round(deltas / (0.25 * beat_duration)) * (0.25 * beat_duration)
        
        # Simple coherence metric: variance of quantized deltas
        coherence = 1 / (np.var(quantized_deltas) + 1e-5)
        print(f"  Rhythm lattice coherence: {coherence:.2f}")
    else:
        print("  No onsets detected.")
    
    # Extract features at onsets for further mapping (ensures no index errors)
    cqt_at_onsets = S_norm[:, onset_frames]
    
    print(f"  Analyzed {file} successfully. Number of onsets: {len(onset_frames)}")

def generate_synthetic_signals():
    sr = 22050
    duration = 5
    t = np.linspace(0, duration, int(sr * duration))
    # Synthetic birdsong-like (high freq)
    y_bird = np.sin(2 * np.pi * 5000 * t) * (np.random.rand(len(t)) > 0.9)
    # Synthetic orchestra-like (mid)
    y_orch = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    # Synthetic rock-like (with beat)
    y_rock = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 200 * t) * (t % 0.5 < 0.25)
    return [
        ('synthetic_birdsong', y_bird, sr),
        ('synthetic_orchestra', y_orch, sr),
        ('synthetic_rock', y_rock, sr)
    ]

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    available_files = [f for f in wav_files if os.path.exists(f)]
    
    if not available_files:
        print("No WAV files available. Falling back to synthetic signals.")
        signals = generate_synthetic_signals()
    else:
        print("Analyzing available WAV files.")
        signals = []
        for file in available_files:
            y, sr = librosa.load(file, sr=22050)
            signals.append((file, y, sr))
    
    for name, y, sr in signals:
        print(f"Analysis for {name}:")
        try:
            analyze_audio(name, y, sr)
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
        print("---")

if __name__ == "__main__":
    main()