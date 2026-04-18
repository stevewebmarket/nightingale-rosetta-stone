# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed Formatting Errors + Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

# Constants for analysis
SAMPLE_RATE = 22050
CQT_N_BINS = 96  # Standardized for CQT invariance across sounds
CQT_BINS_PER_OCTAVE = 12
HOP_LENGTH = 512
ONSET_BACKTRACK = True

# Available WAV files
WAV_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def detect_sound_type(y, sr):
    """Detect sound type based on spectral centroid."""
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    if centroid > 5000:
        return 'high-centroid', 0.01  # Sensitive for birdsong-like
    elif centroid > 2000:
        return 'mid-centroid', 0.05  # Balanced for orchestral/rock
    else:
        return 'low-centroid', 0.1  # Relaxed for bass-heavy

def analyze_audio(file_path):
    """Analyze a single audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        sound_type, onset_delta = detect_sound_type(y, sr)
        print(f"Analysis for {os.path.basename(file_path)}:")
        print(f"  Detected {sound_type} sound (e.g., {'birdsong' if 'high' in sound_type else 'orchestral or rock' if 'mid' in sound_type else 'bass-heavy'}), using {'sensitive' if 'high' in sound_type else 'balanced' if 'mid' in sound_type else 'relaxed'} onset detection.")

        # Compute CQT with fixed n_bins for invariance
        cqt = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCTAVE)
        print(f"  CQT shape: {cqt.shape}, n_bins: {CQT_N_BINS}")

        # Improved onset detection for rhythm lattice
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=ONSET_BACKTRACK, delta=onset_delta)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=HOP_LENGTH)

        # Tempo estimation for coherence
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)

        # Rhythm lattice: Compute inter-onset intervals and coherence metric
        iois = np.diff(onset_times)
        ioi_mean = np.mean(iois) if len(iois) > 0 else 0.0
        coherence = np.std(iois) / ioi_mean if ioi_mean > 0 else 0.0

        # Print results with proper scalar formatting
        print(f"  Detected onsets: {len(onsets)}, mean IOI: {ioi_mean:.2f} s, rhythm coherence: {coherence:.2f}")
        print(f"  Estimated tempo: {float(tempo):.2f} BPM, beats detected: {len(beats)}")  # Ensure scalar

        # Broad sound handling: Additional features
        rms = np.mean(librosa.feature.rms(y=y))
        print(f"  RMS energy: {rms:.4f}")

    except Exception as e:
        print(f"  Error analyzing {os.path.basename(file_path)}: {str(e)}")

def main():
    print("Analyzing available WAV files.")
    for wav in WAV_FILES:
        if os.path.exists(wav):
            analyze_audio(wav)
        else:
            print(f"File {wav} not found, skipping.")

    if not WAV_FILES:
        print("No WAV files available, generating synthetic test signal.")
        # Synthetic signal fallback (e.g., sine wave)
        duration = 5.0
        sr = SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        # Analyze synthetic
        analyze_audio('synthetic.wav')  # Dummy name for printing

if __name__ == "__main__":
    main()