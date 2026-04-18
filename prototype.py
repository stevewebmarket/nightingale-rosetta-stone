# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Coherence + CQT Invariance + Broad Handling
# =============================================================================

import librosa
import numpy as np
import os

# Constants
SAMPLE_RATE = 22050
AVAILABLE_FILES = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

def compute_cqt_invariance(y, sr):
    """Compute CQT with invariance enhancements."""
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
    # Normalize for invariance to amplitude
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    # Simple invariance to pitch shift by averaging across octaves (placeholder)
    octave_avg = np.mean(cqt_mag.reshape(-1, 12, cqt_mag.shape[1]), axis=1)
    return octave_avg

def compute_rhythm_lattice(y, sr):
    """Improved rhythm lattice using tempogram and onset strength."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    # Enhance lattice: compute autocorrelation for coherence
    auto = librosa.autocorrelate(onset_env)
    # Simple rhythm lattice as a grid of tempogram features
    lattice = np.outer(auto[:10], tempogram.mean(axis=1)[:10])  # 10x10 lattice placeholder
    return lattice, tempo

def compute_coherence(y, sr):
    """Compute signal coherence using MFCC correlation."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    corr_matrix = np.corrcoef(mfcc)
    coherence_score = np.mean(corr_matrix)
    return coherence_score

def analyze_audio(file_path):
    """Analyze a single audio file with broad sound handling."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Broad handling: normalize audio
        y = librosa.util.normalize(y)
        
        # Compute features
        cqt_inv = compute_cqt_invariance(y, sr)
        rhythm_lattice, tempo = compute_rhythm_lattice(y, sr)
        coherence = compute_coherence(y, sr)
        
        # Output summary
        print(f"Analysis for {file_path}:")
        print(f"  Tempo: {tempo}")
        print(f"  Coherence Score: {coherence}")
        print(f"  CQT Invariance Shape: {cqt_inv.shape}")
        print(f"  Rhythm Lattice Shape: {rhythm_lattice.shape}")
        print("---")
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")

def main():
    print("Analyzing available WAV files.")
    for file in AVAILABLE_FILES:
        analyze_audio(file)

if __name__ == "__main__":
    main()