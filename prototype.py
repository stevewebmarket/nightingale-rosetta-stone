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

# List of available WAV files in the current working directory
available_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']

# Function to analyze a single audio file
def analyze_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Normalize audio to improve handling for broad sounds (low RMS cases like birdsong)
        y = y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y
        
        # Estimated tempo using updated function (addresses FutureWarning)
        tempo = librosa.feature.rhythm.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
        
        # Mean autocorrelation value
        autocorr = librosa.autocorrelate(y)
        mean_autocorr = np.mean(autocorr)
        
        # Chroma features with CQT for invariance (enhanced with hop_length for better resolution)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        chroma_shape = chroma.shape
        
        # Average RMS (normalized)
        rms = librosa.feature.rms(y=y)
        avg_rms = np.mean(rms) / np.max(rms) if np.max(rms) != 0 else 0
        
        # Improved rhythm lattice: Use tempogram for dominant tempo to avoid inf issues
        # Compute tempogram for rhythm lattice representation
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_envelope, sr=sr)
        
        # Find dominant tempo by averaging the tempogram and finding peak
        tempogram_mean = np.mean(tempogram, axis=1)
        peak_idx = np.argmax(tempogram_mean)
        tempos = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
        dominant_tempo = tempos[peak_idx] if peak_idx > 0 else tempo[0]  # Fallback to estimated tempo if peak at 0
        
        # Improved feature coherence metric: Correlation between chroma and tempogram for better coherence
        # Reshape for correlation
        chroma_mean = np.mean(chroma, axis=0)
        tempogram_resized = np.mean(tempogram, axis=0)[:len(chroma_mean)]
        chroma_mean = chroma_mean[:len(tempogram_resized)]
        if len(chroma_mean) > 1 and len(tempogram_resized) > 1:
            coherence = np.corrcoef(chroma_mean, tempogram_resized)[0, 1]
        else:
            coherence = 0.0
        
        # Print results for this file
        print(f"Analyzing {os.path.basename(file_path)}")
        print(f"Estimated tempo for {os.path.basename(file_path)}: {tempo} BPM")
        print(f"Mean autocorrelation value for {os.path.basename(file_path)}: {mean_autocorr}")
        print(f"Chroma shape for {os.path.basename(file_path)} (CQT invariant): {chroma_shape}")
        print(f"Average RMS for {os.path.basename(file_path)} (normalized): {avg_rms}")
        print(f"Dominant tempo from rhythm lattice for {os.path.basename(file_path)}: {dominant_tempo} BPM")
        print(f"Feature coherence metric for {os.path.basename(file_path)}: {coherence}")
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

# Main execution
if __name__ == "__main__":
    print("Analyzing available WAV files.")
    for file in available_files:
        if os.path.exists(file):
            analyze_audio(file)
        else:
            print(f"File not found: {file}")
    if not available_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example (sine wave)
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=1)
        analyze_audio('synthetic.wav')  # Placeholder name for synthetic