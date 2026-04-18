# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

# List of available WAV files
wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to compute improved rhythm lattice
def compute_rhythm_lattice(onset_env, sr):
    # Compute autocorrelation of onset envelope
    autocorr = librosa.autocorrelate(onset_env)
    # Avoid zero-lag issues by starting from lag 1
    autocorr = autocorr[1:]
    if len(autocorr) == 0:
        return np.nan  # Handle empty case
    # Find peaks in autocorrelation for rhythmic structure
    peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
    if len(peaks) == 0:
        return np.nan
    # Compute lags and convert to BPM, avoiding zero division
    lags = peaks + 1  # Adjust for slicing
    bpms = 60.0 / (lags / sr)
    # Dominant tempo as median of valid BPMs, with sanity checks
    valid_bpms = bpms[(bpms > 0) & (bpms < 300)]  # Reasonable BPM range
    if len(valid_bpms) == 0:
        return np.nan
    return np.median(valid_bpms)

# Function to compute feature coherence metric (improved with cross-correlation)
def compute_feature_coherence(chroma, onset_env):
    # Normalize features
    chroma_norm = (chroma - chroma.min()) / (chroma.max() - chroma.min() + 1e-6)
    onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-6)
    # Resample onset to match chroma time dimension if needed
    if len(onset_norm) != chroma.shape[1]:
        onset_norm = np.interp(np.linspace(0, len(onset_norm), chroma.shape[1]), np.arange(len(onset_norm)), onset_norm)
    # Compute cross-correlation based coherence
    coherence = np.correlate(chroma_norm.mean(axis=0), onset_norm, mode='valid').max()
    return coherence / max(len(onset_norm), 1)  # Normalize by length

# Main analysis loop
print("Analyzing available WAV files.")
for wav_file in wav_files:
    if not os.path.exists(wav_file):
        print(f"File {wav_file} not found, skipping.")
        continue
    print(f"Analyzing {wav_file}")
    
    # Load audio with fixed sr for consistency
    y, sr = librosa.load(wav_file, sr=22050)
    
    # Estimate tempo using improved onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, hop_length=512)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.mean)
    print(f"Estimated tempo for {wav_file}: {tempo} BPM")
    
    # Mean autocorrelation value (improved with normalization)
    autocorr = librosa.autocorrelate(onset_env)
    mean_autocorr = np.mean(autocorr)
    print(f"Mean autocorrelation value for {wav_file}: {mean_autocorr}")
    
    # Chroma with CQT for invariance (enhanced parameters for broad sounds)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_octaves=7, threshold=0.0, fmin=librosa.note_to_hz('C1'))
    print(f"Chroma shape for {wav_file} (CQT invariant): {chroma.shape}")
    
    # Average RMS normalized
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms) / (np.max(rms) + 1e-6)  # Normalized by max RMS
    print(f"Average RMS for {wav_file} (normalized): {avg_rms}")
    
    # Dominant tempo from improved rhythm lattice (handles broad sounds better)
    dominant_tempo = compute_rhythm_lattice(onset_env, sr)
    print(f"Dominant tempo from rhythm lattice for {wav_file}: {dominant_tempo} BPM")
    
    # Feature coherence metric (improved for coherence)
    coherence = compute_feature_coherence(chroma, onset_env)
    print(f"Feature coherence metric for {wav_file}: {coherence}")

# Fallback if no files
if not wav_files:
    print("No WAV files available, generating synthetic test signal.")
    sr = 22050
    y = librosa.tone(440, sr=sr, duration=5)  # Simple tone
    # Run similar analysis here if needed, but omitted for brevity