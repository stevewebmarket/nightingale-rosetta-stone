# =============================================================================
# Nightingale Mapping – Rosetta Stone Prototype v13.0
# Originator: Stephen OConnor (@nightingalemap) – The Nightingale Mapping
# Date: April 17, 2026
# Live Hub: https://github.com/stevewebmarket/nightingale-rosetta-stone
# FULL SELF-ITERATING SWARM MODE ACTIVATED
# =============================================================================

import numpy as np
from copy import deepcopy

print("✅ Nightingale Mapping Rosetta Stone v13.0 – FULL SELF-ITERATING SWARM LAUNCHED\n")

sr = 44100
duration = 1.0

def generate_tone(freq, dur=duration, sample_rate=sr):
    t = np.linspace(0, dur, int(sample_rate * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def normalize_audio(sound):
    return sound / (np.max(np.abs(sound)) + 1e-8)

def pitch_shift(sound, semitones=5):
    factor = 2 ** (semitones / 12.0)
    indices = np.arange(len(sound)) * factor
    indices = indices[indices < len(sound)]
    shifted = np.interp(indices, np.arange(len(sound)), sound)
    return np.pad(shifted, (0, len(sound) - len(shifted)), 'constant')[:len(sound)]

def spectrogram_correlation(sound1, sound2):
    spec1 = np.abs(np.fft.rfft(sound1))
    spec2 = np.abs(np.fft.rfft(sound2[:len(spec1)]))
    min_len = min(len(spec1), len(spec2))
    if min_len == 0: return 0.0
    c = np.corrcoef(spec1[:min_len], spec2[:min_len])[0, 1]
    return float(max(min(c, 1.0), 0.0))

def build_sound_rep(sound):
    sound = normalize_audio(sound)
    fft = np.abs(np.fft.rfft(sound))
    freqs = np.fft.rfftfreq(len(sound), 1/sr)
    peak_idx = np.argsort(fft)[-12:][::-1]
    peak_freqs = [float(freqs[i]) for i in peak_idx if freqs[i] > 20]
    return {"dominant_freq": round(peak_freqs[0], 2) if peak_freqs else 0.0,
            "peak_freqs": [round(f, 2) for f in peak_freqs]}

def spectral_entropy(sound):
    spec = np.abs(np.fft.rfft(normalize_audio(sound)))
    p = spec / (np.sum(spec) + 1e-8)
    ent = -np.sum(p * np.log2(p + 1e-8))
    return float(ent / np.log2(len(p) + 1e-8))

def consonance_bonus(sound, tol=0.012):
    targets = [1.25, 1.3333, 1.5, 1.6667, 2.0]
    peak_freqs = build_sound_rep(sound)["peak_freqs"]
    if len(peak_freqs) < 2: return 0.0
    score = 0.0
    count = 0
    for i in range(len(peak_freqs)):
        for j in range(i+1, len(peak_freqs)):
            ratio = peak_freqs[j] / peak_freqs[i]
            for tr in targets:
                if abs(ratio - tr) < tol:
                    score += 1.0
                    count += 1
                    break
    return min(score / max(count, 1), 1.0) if count > 0 else 0.0

def harmonic_coherence(sound):
    ent = spectral_entropy(sound)
    cons = consonance_bonus(sound)
    return float(0.52 * (1 - ent) + 0.48 * cons)

def fidelity_score(relation=0.0, coherence=0.0, invariance=0.0, compress=0.0, novelty=0.0):
    return 0.28*relation + 0.22*coherence + 0.25*invariance + 0.15*compress + 0.10*novelty

# Self-iterating Rhythm Lattice (injected from your 1.25/1.5 chord + real consonance=1.0 phrases)
def rhythm_lattice_encode(m, p):
    ratios = m.get("values", [1.0, 1.25, 1.5, 2.0])
