import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf

# Function to compute improved CQT with invariance enhancements
def compute_cqt(y, sr):
    # Use librosa CQT for log-frequency resolution, promoting pitch invariance
    cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
    # Normalize for better invariance to amplitude variations
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    return cqt_mag

# Function for rhythm lattice: structured grid for beat patterns
def rhythm_lattice(beats, tempo, sr, length):
    # Create a lattice grid for rhythms, improving coherence by aligning to tempo
    t = np.arange(length) / sr
    lattice = np.zeros_like(t)
    beat_interval = 60 / tempo  # seconds per beat
    for beat in beats:
        pos = int(beat * sr)
        if pos < length:
            lattice[pos] = 1
    # Interpolate sub-beats for finer lattice (e.g., quarter notes)
    sub_beats = np.arange(0, len(t) / sr, beat_interval / 4)
    for sb in sub_beats:
        pos = int(sb * sr)
        if pos < length:
            lattice[pos] = 0.5  # Weaker pulses for sub-divisions
    return lattice

# Main prototype function
def main():
    # Generate synthetic audio for broad sound handling (mix of tones, noise, beats)
    sr = 22050
    duration = 10  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Broad sounds: sine waves, noise, rhythmic elements
    freq1 = 440  # A4
    freq2 = 660  # Approx E5 for harmony
    tone1 = 0.5 * np.sin(2 * np.pi * freq1 * t)
    tone2 = 0.3 * np.sin(2 * np.pi * freq2 * t)
    noise = 0.1 * np.random.randn(len(t))  # White noise for broad handling
    rhythmic_pulse = 0.2 * np.sin(2 * np.pi * 120/60 * t)  # Simulated beat at 120 BPM
    
    y = tone1 + tone2 + noise + rhythmic_pulse  # Combined signal
    
    # Compute CQT for analysis, with invariance
    cqt = compute_cqt(y, sr)
    print("CQT shape:", cqt.shape)  # Debug
    
    # Beat tracking for rhythm detection, improved for coherence
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512, tightness=100)
    print(f"Estimated tempo: {tempo}")
    
    # Generate rhythm lattice for better structure
    y_length = len(y)
    y_beat = rhythm_lattice(beats, tempo, sr, y_length)
    
    # Smooth pulses with Gaussian for realism, fixing the import issue
    gauss_window = signal.windows.gaussian(200, std=10)
    y_beat = signal.convolve(y_beat, gauss_window, mode='same')
    
    # Normalize
    y_beat /= np.max(np.abs(y_beat)) + 1e-8
    
    # Mix with original for coherent output
    output = y + 0.5 * y_beat  # Enhance rhythm coherence
    
    # Handle broad sounds: apply dynamic range compression for various input types
    output = librosa.util.normalize(output)
    
    # Save output
    sf.write('output.wav', output, sr)
    print("Output saved as output.wav")

if __name__ == "__main__":
    main()