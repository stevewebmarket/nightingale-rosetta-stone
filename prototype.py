# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Enhanced CQT Invariance and Rhythm Lattice Refinement
# =============================================================================

import librosa
import numpy as np

def analyze_audio(files):
    for file in files:
        print(f"Analysis for {file}:")
        y, sr = librosa.load(file, sr=22050)

        # Spectral centroid to detect type
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = np.mean(centroid)

        is_mid_centroid = mean_centroid > 1500
        if is_mid_centroid:
            print("  Detected mid-centroid sound.")
            fmin = librosa.note_to_hz('C3')
            bins_per_octave = 64
        else:
            print("  Detected low-centroid sound.")
            fmin = librosa.note_to_hz('C1')
            bins_per_octave = 48

        n_bins = 384

        # Onsets
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        print(f"  Detected onsets: {len(onset_times)}")

        # Rhythm metrics
        if len(onset_times) > 1:
            iois = np.diff(onset_times)
            mean_ioi = np.mean(iois)
            rhythm_coherence = 1 - (np.std(iois) / mean_ioi)
            print(f"  mean IOI: {mean_ioi:.2f} s, rhythm coherence: {rhythm_coherence:.2f}")
        else:
            print("  Insufficient onsets for IOI calculation.")
            continue

        # Rhythm lattice using improved tempo estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        lattice_base = (60 / tempo) / 12  # Refined subdivision for better lattice fit
        print(f"  Rhythm lattice base: {lattice_base:.3f} s")

        # Lattice coherence: improved metric based on onset alignment to lattice
        mods = onset_times % lattice_base
        dists = np.minimum(mods, lattice_base - mods)
        mean_dist_norm = np.mean(dists) / (lattice_base / 2)
        lattice_coherence = 1 - mean_dist_norm
        print(f"  lattice coherence: {lattice_coherence:.2f}")

        # CQT with adaptive parameters for improved invariance and broad sound handling
        cqt = librosa.cqt(y=y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
        print(f"  CQT shape: {cqt.shape}, n_bins: {cqt.shape[0]}")

        # Improved CQT shift invariance metric: average correlation after 1-bin shift
        cqt_abs = np.abs(cqt)
        shifted = np.roll(cqt_abs, 1, axis=0)
        corrs = []
        for t in range(cqt_abs.shape[1]):
            if np.sum(cqt_abs[:, t]) > 0:
                corr = np.corrcoef(cqt_abs[:, t], shifted[:, t])[0, 1]
                corrs.append(corr)
        invariance_metric = np.mean(corrs) if corrs else 0.0
        print(f"  CQT shift invariance metric: {invariance_metric:.2f} (higher is more invariant)")
        print()  # Blank line for separation

if __name__ == "__main__":
    files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not files:
        # Fallback to synthetic test signals if no files
        print("No WAV files available. Generating synthetic test signals.")
        sr = 22050
        t = np.linspace(0, 10, 10 * sr)
        y_sine = np.sin(440 * 2 * np.pi * t)  # Simple sine wave
        y_noise = np.random.randn(len(t)) * 0.5  # Noise
        # Save as temp files or analyze directly (here, simulate files)
        synthetic_files = ['synthetic_sine.wav', 'synthetic_noise.wav']
        # For prototype, just analyze in-memory
        analyze_audio([])  # Placeholder, but actually would process y_sine, etc.
    else:
        analyze_audio(files)