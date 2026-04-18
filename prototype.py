# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Fixed scalar conversion error + Enhanced rhythm lattice, coherence, CQT invariance, broad sound handling
# =============================================================================

import os
import numpy as np
import librosa

def analyze_audio(file):
    try:
        y, sr = librosa.load(file, sr=22050)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        if centroid < 1000:
            sound_type = "low-centroid sound (e.g., bass-heavy)"
            onset_params = {'backtrack': True, 'delta': 0.05, 'pre_max': 0.03, 'post_max': 0.03}
        elif centroid < 4000:
            sound_type = "mid-centroid sound (e.g., orchestral or rock)"
            onset_params = {'backtrack': False, 'delta': 0.1, 'pre_max': 0.02, 'post_max': 0.02}
        else:
            sound_type = "high-centroid sound (e.g., birdsong)"
            onset_params = {'backtrack': True, 'delta': 0.07, 'pre_max': 0.04, 'post_max': 0.04}
        
        print(f"  Detected {sound_type}, using adjusted onset detection.")
        
        # CQT with improved invariance: log amplitude and normalization
        cqt = librosa.cqt(y, sr=sr, hop_length=512, n_bins=96)
        cqt_mag = librosa.amplitude_to_db(np.abs(cqt))
        # Normalize per bin for shift invariance
        cqt_mag -= np.min(cqt_mag, axis=1, keepdims=True)
        cqt_mag /= np.max(cqt_mag, axis=1, keepdims=True) + 1e-6
        
        print(f"  CQT shape: {cqt_mag.shape}, n_bins: {cqt_mag.shape[0]}")
        
        # Onset detection with adaptive parameters
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, **onset_params)
        
        print(f"  Detected onsets: {len(onsets)}", end="")
        
        if len(onsets) < 2:
            print(", insufficient onsets for rhythm analysis.")
            return
        
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=512)
        IOIs = np.diff(onset_times)
        mean_IOI = float(np.mean(IOIs))  # Ensure scalar
        
        print(f", mean IOI: {mean_IOI:.2f} s", end="")
        
        # Improved coherence: autocorrelation with regularization
        if len(IOIs) < 2:
            coherence = 0.0
        else:
            corr_matrix = np.corrcoef(IOIs[:-1], IOIs[1:])
            coherence = corr_matrix[0, 1].item()  # Extract scalar safely
        
        print(f", rhythm coherence: {coherence:.2f}")
        
        # Improved rhythm lattice: estimate base unit from tempo, quantize, compute fit
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]  # Scalar
        base_unit = 60 / tempo / 4  # e.g., sixteenth note assumption
        quantized = np.round(IOIs / base_unit) * base_unit
        lattice_error = np.mean(np.abs(IOIs - quantized)) / mean_IOI
        lattice_coherence = 1 - np.clip(lattice_error, 0, 1)
        
        print(f"  Rhythm lattice base: {base_unit:.3f} s, lattice coherence: {lattice_coherence:.2f}")
        
        # Additional invariance metric: mean pitch shift invariance (simplified)
        bin_shifts = np.diff(np.argmax(cqt_mag, axis=0))
        shift_variance = float(np.std(bin_shifts))  # Ensure scalar
        print(f"  CQT shift invariance metric: {shift_variance:.2f} (lower is more invariant)")
    
    except Exception as e:
        print(f"  Error analyzing {file}: {str(e)}")

if __name__ == "__main__":
    print("Analyzing available WAV files.")
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    synthetic_used = False
    
    for file in files:
        if os.path.exists(file):
            print(f"Analysis for {file}:")
            analyze_audio(file)
        else:
            print(f"{file} not found.")
            synthetic_used = True
    
    if synthetic_used or not files:
        # Fallback to synthetic test signals
        print("No WAV files found, using synthetic test signals.")
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Synthetic 1: Sine wave rhythm
        y_synth1 = np.sin(2 * np.pi * 440 * t) * (t % 0.5 < 0.05)
        librosa.output.write_wav('synth1.wav', y_synth1, sr)
        print("Analysis for synth1.wav (sine rhythm):")
        analyze_audio('synth1.wav')
        
        # Synthetic 2: Noise bursts
        y_synth2 = np.random.randn(len(t)) * (t % 0.3 < 0.02)
        librosa.output.write_wav('synth2.wav', y_synth2, sr)
        print("Analysis for synth2.wav (noise bursts):")
        analyze_audio('synth2.wav')