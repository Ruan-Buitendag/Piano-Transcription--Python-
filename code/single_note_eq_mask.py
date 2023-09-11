import os

import librosa.core

import STFT
import numpy as np

piano_W = "AkPnBsdf"
piano_H = "AkPnBcht"

path_maps = "../MAPS"

midi_note = "59"

midi_note_2 = "65"

duration = 3


def NoteSpectrogram(midi, piano):
    path = "{}/{}/ISOL/NO/MAPS_ISOL_NO_M_S1_M{}_{}.wav".format(path_maps, piano, midi, piano)
    if not os.path.isfile(path):
        path = "{}/{}/ISOL/NO/MAPS_ISOL_NO_M_S0_M{}_{}.wav".format(path_maps, piano, midi, piano)

    spec = STFT.STFT(path, time=10, num_bins=4096)
    mag_spec = spec.get_magnitude_spectrogram()
    start = spec.getDelayIndex()
    mag_spec = mag_spec[:, start:start + round(duration // 0.02)]
    return mag_spec


def ScaleSpectrogram(spectrogram, midi_note, midi_note_2):
    factor = librosa.core.midi_to_hz(int(midi_note_2)) / librosa.core.midi_to_hz(int(midi_note))

    target_bins = range(target.shape[0])
    eq_bins = range(eq.shape[0])

    scaled_eq = factor * eq_bins

    scaled_eq = np.round(scaled_eq)

    aa = np.where(scaled_eq > target.shape[0])
    scaled_eq = np.delete(scaled_eq, aa)

    new_eq = np.zeros(target.shape)

    for i in range(eq.shape[1]):
        new_eq[:, i] = np.interp(target_bins, scaled_eq, eq[:scaled_eq.shape[0], i])

    return new_eq


def ScaleSpectrogramNoStretch(spectrogram, midi_note, midi_note_2):
    original_hz = librosa.core.midi_to_hz(int(midi_note))
    original_fundamental_bin = int(np.round(original_hz / 22050 * 4096))

    new_hz = librosa.core.midi_to_hz(int(midi_note_2))
    new_fundamental_bin = int(np.round(new_hz / 22050 * 4096))

    scaled_eq = np.zeros(spectrogram.shape)

    for harmonic in range(10):
        original_harmonic_bin = harmonic * original_fundamental_bin
        new_harmonic_bin = harmonic * new_fundamental_bin + 1

        if original_harmonic_bin > 4096 or new_harmonic_bin > 4096:
            break

        if new_harmonic_bin >= 10:
            scaled_eq[new_harmonic_bin - 10:new_harmonic_bin + 10, :] = spectrogram[
                                                                        original_harmonic_bin - 10:original_harmonic_bin + 10,
                                                                        :]

    return scaled_eq


def CalculateEQMask(midi_note, piano_H, piano_W, threshold=0.1):
    H_mag_spec = NoteSpectrogram(midi_note, piano_H)
    W_mag_spec = NoteSpectrogram(midi_note, piano_W)

    difference = H_mag_spec - W_mag_spec
    eq = difference / np.max(W_mag_spec)

    eq[(eq < threshold) & (eq > -threshold)] = 0

    return eq


def TemplateEQ(eq):
    chopped_eq = eq[:, :30]

    template_eq = np.mean(chopped_eq, axis=1)

    return template_eq

def TemplateEQWithTime(eq):
    # Calculate the window size
    eq = eq[:, :40]

    num_columns = eq.shape[1]
    window_size = num_columns // 10

    # Initialize an empty result array
    template_eq = np.zeros((eq.shape[0], 10))

    # Calculate the mean over the windows and store in the result array
    for i in range(10):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        template_eq[:, i] = np.mean(eq[:, window_start:window_end], axis=1)

    return template_eq


eq = CalculateEQMask(midi_note, piano_H, piano_W)

target = NoteSpectrogram(midi_note_2, piano_H)

scaled_eq = ScaleSpectrogram(eq, midi_note, midi_note_2)
better_scaled_eq = ScaleSpectrogramNoStretch(eq, midi_note, midi_note_2)

template_eq = TemplateEQ(better_scaled_eq).T

H_mag_spec = NoteSpectrogram(midi_note, piano_H)

W_mag_spec = NoteSpectrogram(midi_note, piano_W)

H_prime = W_mag_spec + eq * np.mean(W_mag_spec)

difference = H_mag_spec - H_prime

a = 0
