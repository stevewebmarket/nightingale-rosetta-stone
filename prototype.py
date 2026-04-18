# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Improved Rhythm Lattice, Coherence, CQT Invariance, Broad Sound Handling
# =============================================================================

import librosa
import numpy as np
import os

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    available_files = [f for f in wav_files if os.path.exists(f)]
    
    if not available_files:
        # Fall back to synthetic test signals
        sr = 22050
        # Synthetic birdsong-like: high-frequency chirps
        t = np.linspace(0, 5, 5 * sr)
        y_bird = np.sin(2 * np.pi * 4000 * t) * (np.sin(2 * np.pi * 5 * t) > 0.5)
        # Synthetic orchestra-like: harmonic tones
        y_orch = np.sum([np.sin(2 * np.pi * freq * t) for freq in [220, 330, 440]], axis=0) / 3
        # Synthetic rock-like: beat with tones
        y_rock = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t) * (t % 0.25 < 0.125)
        analyses = [
            ('synthetic_birdsong', y_bird),
            ('synthetic_orchestra', y_orch),
            ('synthetic_rock', y_rock)
        ]
    else:
        analyses = [(file, None) for file in available_files]
    
    print("Analyzing available WAV files." if available_files else "No WAV files available, using synthetic test signals.")
    
    for name, y_preload in analyses:
        print(f'Analysis for {name}:')
        if y_preload is None:
            y, sr = librosa.load(name, sr=22050)
        else:
            y = y_preload
            sr = 22050
        
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        if centroid > 4000:
            sound_type = 'high-centroid'
            example = 'birdsong'
            n_bins = 120
            bins_per_octave = 24
            delta = 0.05
        elif centroid > 1000:
            sound_type = 'mid-centroid'
            example = 'orchestral or rock'
            n_bins = 96
            bins_per_octave = 12
            delta = 0.1
        else:
            sound_type = 'low-centroid'
            example = 'bass-heavy'
            n_bins = 72
            bins_per_octave = 12
            delta = 0.15
        
        print(f'  Detected {sound_type} sound (e.g., {example}), using adjusted onset detection.')
        
        # First pass: estimate tempo and IOI with default settings
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        mean_beat = np.mean(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
        
        onsets_first = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, delta=delta)
        onset_times_first = librosa.frames_to_time(onsets_first, sr=sr)
        iois_first = np.diff(onset_times_first)
        mean_ioi = np.mean(iois_first) if len(iois_first) > 0 else mean_beat
        
        # Set dynamic hop_length for better resolution
        target_resolution = min(mean_ioi, mean_beat) / 32
        hop_length = max(16, int(sr * target_resolution))  # Ensure not too small
        
        # Compute CQT with improved parameters
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=librosa.note_to_hz('C1'),
                          n_bins=n_bins, bins_per_octave=bins_per_octave)
        print(f'  CQT shape: {cqt.shape}, n_bins: {n_bins}')
        
        # Refined onset detection with custom hop_length
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True, delta=delta)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)
        detected_onsets = len(onsets)
        iois = np.diff(onset_times)
        mean_ioi = np.mean(iois) if len(iois) > 0 else 0
        
        # Improved rhythm coherence: exponential decay of coefficient of variation
        if len(iois) > 1:
            cv = np.std(iois) / mean_ioi
            rhythm_coherence = np.exp(-cv * 2)  # Adjusted for sensitivity, 0 to 1
        else:
            rhythm_coherence = 0
        
        # Improved rhythm lattice base: subdivision of beat
        lattice_base = mean_beat / 4 if mean_beat > 0 else target_resolution
        
        # Improved lattice coherence: alignment error
        if detected_onsets > 1:
            snapped = np.round(onset_times / lattice_base) * lattice_base
            errors = np.abs(onset_times - snapped) / lattice_base
            lattice_coherence = max(0, 1 - np.mean(errors) * 2)  # Scaled for better range
        else:
            lattice_coherence = 0
        
        # Improved CQT shift invariance metric: using log magnitude, frequency shift
        mag = np.abs(cqt)
        log_mag = np.log1p(mag)
        shifted = np.roll(log_mag, 1, axis=0)  # Shift by one bin (approx semitone)
        diff = np.linalg.norm(log_mag - shifted, ord='fro') / np.linalg.norm(log_mag, ord='fro')
        invariance_metric = diff
        
        print(f'  Detected onsets: {detected_onsets}, mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}')
        print(f'  Rhythm lattice base: {lattice_base:.3f} s, lattice coherence: {lattice_coherence:.2f}')
        print(f'  CQT shift invariance metric: {invariance_metric:.2f} (lower is more invariant)')

if __name__ == '__main__':
    main()