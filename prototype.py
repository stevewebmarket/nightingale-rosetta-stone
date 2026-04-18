# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced Rhythm Lattice + Coherence + CQT Invariance + Broad Handling
# =============================================================================

import os
import librosa
import numpy as np

def detect_sound_type(y, sr):
    """Detect sound type based on spectral centroid."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_centroid = np.mean(centroid)
    if mean_centroid < 1000:
        return "low-centroid sound (e.g., ambient)", "loose"
    elif mean_centroid < 3000:
        return "mid-centroid sound (e.g., orchestral)", "balanced"
    else:
        return "high-centroid sound (e.g., birdsong)", "strict"

def analyze_rhythm(y, sr, onset_strength, hop_length=512):
    """Analyze rhythm using tempogram for lattice structure."""
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_strength, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_strength, sr=sr, hop_length=hop_length)
    rhythm_lattice = np.mean(tempogram, axis=1)  # Simplified lattice as mean across time
    return tempo, len(beats), rhythm_lattice

def enhanced_onset_detection(y, sr, mode, hop_length=512):
    """Enhanced onset detection with backtracking and delta adjustments."""
    if mode == "loose":
        delta = 0.05
        backtrack = True
    elif mode == "balanced":
        delta = 0.1
        backtrack = True
    else:  # strict
        delta = 0.2
        backtrack = False
    
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr, hop_length=hop_length, backtrack=backtrack, delta=delta)
    return onsets, onset_strength

def compute_cqt_invariance(y, sr, hop_length=512):
    """Compute CQT for pitch invariance."""
    cqt = librosa.cqt(y, sr=sr, hop_length=hop_length)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
    invariant_features = np.mean(cqt_mag, axis=1)  # Mean across time for invariance
    return invariant_features

def improve_coherence(onsets, y, sr):
    """Improve coherence by filtering onsets based on energy."""
    rms = librosa.feature.rms(y=y)
    coherent_onsets = [o for o in onsets if rms[0, int(o * (sr / 512))] > 0.01]  # Threshold for coherence
    return coherent_onsets

def analyze_file(filename):
    try:
        y, sr = librosa.load(filename, sr=22050)
        sound_type, onset_mode = detect_sound_type(y, sr)
        print(f"  Detected {sound_type}, using {onset_mode} onset detection.")
        
        onsets, onset_strength = enhanced_onset_detection(y, sr, onset_mode)
        coherent_onsets = improve_coherence(onsets, y, sr)
        print(f"  Onsets detected: {len(coherent_onsets)}")
        
        tempo, beat_count, rhythm_lattice = analyze_rhythm(y, sr, onset_strength)
        print(f"  Estimated tempo: {tempo:.2f} BPM")
        print(f"  Beats detected: {beat_count}")
        print(f"  Rhythm lattice mean: {np.mean(rhythm_lattice):.4f}")
        
        cqt_invariant = compute_cqt_invariance(y, sr)
        print(f"  CQT invariant features mean: {np.mean(cqt_invariant):.4f}")
        
    except Exception as e:
        print(f"  Error analyzing {filename}: {str(e)}")

def main():
    print("Analyzing available WAV files.")
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    
    for filename in wav_files:
        if os.path.exists(filename):
            print(f"Analysis for {filename}:")
            analyze_file(filename)
            print("---")
        else:
            print(f"File {filename} not found. Skipping.")
            print("---")
    
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example
        sr = 22050
        y = librosa.tone(440, sr=sr, duration=5)
        sound_type, onset_mode = detect_sound_type(y, sr)
        print(f"  Synthetic signal: Detected {sound_type}, using {onset_mode} onset detection.")
        onsets, onset_strength = enhanced_onset_detection(y, sr, onset_mode)
        coherent_onsets = improve_coherence(onsets, y, sr)
        print(f"  Onsets detected: {len(coherent_onsets)}")
        
        tempo, beat_count, rhythm_lattice = analyze_rhythm(y, sr, onset_strength)
        print(f"  Estimated tempo: {tempo:.2f} BPM")
        print(f"  Beats detected: {beat_count}")
        print(f"  Rhythm lattice mean: {np.mean(rhythm_lattice):.4f}")
        
        cqt_invariant = compute_cqt_invariance(y, sr)
        print(f"  CQT invariant features mean: {np.mean(cqt_invariant):.4f}")

if __name__ == "__main__":
    main()