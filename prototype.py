import librosa
import numpy as np

print("✅ Nightingale Mapping Rosetta Stone v16.3 – Improved Rhythm Lattice and Invariance")

# Generate test sound: A440 tone
sr = 22050
duration = 1.0
y = librosa.tone(440, sr=sr, duration=duration)

# Compute CQT with higher resolution for better invariance
fmin = librosa.note_to_hz('C1')
bpo = 24  # Increased bins per octave for invariance
n_bins = 168  # Adjusted to cover similar range
cqt = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, hop_length=512))

# Mean spectrum
mean_spec = np.mean(cqt, axis=1)
mean_spec = mean_spec / (np.sum(mean_spec) + 1e-10)  # normalize to probability

# Spec entropy with handling for small values
spec_entropy = -np.sum(mean_spec * np.log(np.maximum(mean_spec, 1e-20))) / np.log(len(mean_spec))

# Dominant freq
freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=bpo)
dominant = freqs[np.argmax(mean_spec)]

# Peak freqs
peak_freqs = [f"{dominant:.1f}"]

# Consonance bonus (1 - normalized entropy)
bonus = 1 - spec_entropy

# Invariance +5st
y_shift = librosa.effects.pitch_shift(y, sr=sr, n_steps=5)
cqt_shift = np.abs(librosa.cqt(y_shift, sr=sr, fmin=fmin, n_bins=n_bins, bins_per_octave=bpo, hop_length=512))
mean_spec_shift = np.mean(cqt_shift, axis=1)
mean_spec_shift = mean_spec_shift / (np.sum(mean_spec_shift) + 1e-10)

# Shift original spec by 5 semitones (5 * (bpo/12) = 5*2 = 10 bins)
shifted_spec = np.roll(mean_spec, -10)

# Invariance as mean difference
inv = np.mean(mean_spec_shift - shifted_spec)

# Rhythm analysis with multi-scale tempogram
hop_length = 512
odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

# Different window lengths for broad rhythm handling
win_short = 192
win = 384
win_long = 768

tempogram_short = librosa.feature.tempogram(onset_envelope=odf, sr=sr, hop_length=hop_length, win_length=win_short)
tempogram = librosa.feature.tempogram(onset_envelope=odf, sr=sr, hop_length=hop_length, win_length=win)
tempogram_long = librosa.feature.tempogram(onset_envelope=odf, sr=sr, hop_length=hop_length, win_length=win_long)

# Pad to max shape for combining (improved lattice)
max_win = max(win_short, win, win_long)
def pad_tempo(t, target):
    pad_width = target - t.shape[0]
    return np.pad(t, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

temp_short_pad = pad_tempo(tempogram_short, max_win)
temp_pad = pad_tempo(tempogram, max_win)
temp_long_pad = pad_tempo(tempogram_long, max_win)

# Average for lattice, normalize each for coherence
temp_short_pad /= (np.max(temp_short_pad) + 1e-10)
temp_pad /= (np.max(temp_pad) + 1e-10)
temp_long_pad /= (np.max(temp_long_pad) + 1e-10)
tempogram_lattice = (temp_short_pad + temp_pad + temp_long_pad) / 3.0

# Coherence as mean of max per frame, handle low energy
coherence = np.mean(np.max(tempogram_lattice, axis=0))
if np.isnan(coherence) or np.all(odf == 0):
    coherence = 0.0

# Print analysis
print("\n--- Analysis: A440 Tone ---")
print(f"Dominant: {dominant:.1f} | Coherence: {coherence:.3f} | Invariance(+5st): {inv:.4f}")
print(f"Peak freqs: {peak_freqs}")
print(f"Consonance bonus: {bonus:.4f}")