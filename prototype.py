# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype.py (permanent name)
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# v16.4 – Dynamic CQT Bins + Improved Rhythm Lattice & Invariance
# =============================================================================

import librosa
import numpy as np
import math
import os

def detect_sound_type(centroid):
    if centroid < 1500:
        return "low-centroid sound (e.g., bass-heavy)", "relaxed onset detection", librosa.note_to_hz('C0')
    elif centroid < 4000:
        return "mid-centroid sound (e.g., orchestral or rock)", "balanced onset detection", librosa.note_to_hz('C1')
    else:
        return "high-centroid sound (e.g., birdsong)", "sensitive onset detection", librosa.note_to_hz('C2')

def analyze_file(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sound_type, onset_type, fmin = detect_sound_type(centroid)
        print(f"  Detected {sound_type}, using {onset_type}.")

        # Dynamic n_bins to respect Nyquist
        bpo = 12  # Base bins per octave, can be adjusted for invariance
        max_f = sr / 2.0
        if fmin >= max_f:
            raise ValueError("fmin must be less than Nyquist frequency.")
        n_bins = math.floor(bpo * math.log2(max_f / fmin)) + 1

        # Compute CQT with improved invariance (log-frequency scaling)
        cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo)

        # Improve coherence: magnitude to db for better dynamic range handling
        cqt_db = librosa.amplitude_to_db(np.abs(cqt))

        # Onset detection with type-specific parameters
        hop_length = 512
        if "relaxed" in onset_type:
            backtrack = False
            pre_post_max = 3
        elif "balanced" in onset_type:
            backtrack = True
            pre_post_max = 5
        else:
            backtrack = True
            pre_post_max = 7

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, S=cqt_db, hop_length=hop_length,
                                                 pre_max=pre_post_max, post_max=pre_post_max)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=backtrack,
                                            pre_max=pre_post_max, post_max=pre_post_max)

        # Improved rhythm lattice: Build a simple lattice from onsets and tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Coherence: Average tempogram for rhythm coherence score
        coherence = np.mean(tempogram)

        print(f"  CQT shape: {cqt.shape}, n_bins: {n_bins}")
        print(f"  Estimated tempo: {tempo:.2f} BPM")
        print(f"  Number of onsets: {len(onsets)}")
        print(f"  Rhythm coherence score: {coherence:.4f}")
        # Broad sound handling: Additional metric for invariance (e.g., octave shift simulation)
        # Simulate shift by transposing CQT (for invariance demo)
        shift_amount = 12  # one octave
        if n_bins > shift_amount:
            shifted_cqt = np.roll(cqt_db, shift_amount, axis=0)
            invariance_diff = np.mean(np.abs(cqt_db - shifted_cqt))
            print(f"  CQT invariance difference (octave shift): {invariance_diff:.4f}")
        else:
            print("  Insufficient bins for invariance shift simulation.")

    except Exception as e:
        print(f"  Error analyzing {file_path}: {str(e)}")

def main():
    wav_files = ['birdsong.wav', 'orchestra.wav', 'rock.wav']
    if not wav_files:
        print("No WAV files available. Falling back to synthetic test signals.")
        # Synthetic signal example
        sr = 22050
        t = np.linspace(0, 5, 5 * sr, endpoint=False)
        y = np.sin(440 * 2 * np.pi * t) + 0.5 * np.sin(880 * 2 * np.pi * t)
        # Analyze synthetic
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        sound_type, onset_type, fmin = detect_sound_type(centroid)
        print(f"  Detected {sound_type} for synthetic signal, using {onset_type}.")
        # Proceed similarly as above, but skip for brevity
        return

    print("Analyzing available WAV files.")
    for file in wav_files:
        if os.path.exists(file):
            print(f"Analysis for {file}:")
            analyze_file(file)
        else:
            print(f"File {file} not found.")

if __name__ == "__main__":
    main()