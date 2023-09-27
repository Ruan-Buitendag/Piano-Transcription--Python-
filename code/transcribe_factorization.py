# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:40:58 2019

"""

import numpy as np
import math
from scipy.signal import find_peaks

from midiutil import MIDIFile


def transcribe_activations(midi_codebook, activation, stft, threshold, sliding_window=5,
                           pourcentage_onset_threshold=0.1, H_normalization=False):
    """
    Transcribe the activation in onset-offset-pitch notes, and format it for txt and midi

    Parameters
    ----------
    midi_codebook: array
        The codebook in midi, for corresponding the activation to their pitch
    activation: array
        The activations find by NMF
    stft: STFT object (see STFT.py)
        The Short Time Fourier Transform of the original signal, used for the sampling rate
    threshold: float
        The threshold for activation to be considered a note
    sliding_window: integer
        The sliding window on which to operate avergaing of activation (minimizing localized peaks misdetection)
        Default: 5
    pourcentage_onset_threshold = 0.1
        Pourcentage of the threshold to consider for resetting the onset.
        The goal is to lower the threshold for a closer onset, as the detection happens after the real onset
        Default: 0.1

    Returns
    -------
    note_tab: list
        Content of transcription_evaluation (onset, offset and midipitch) for txt format (tab with the values in that order)
    MIDI_file_output: MIDIFile
        Content of transcription_evaluation in MIDIFile format.
    """

    presence_of_a_note = False
    note_tab = []
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    # Creation of a .mid file
    MIDI_file_output = MIDIFile(1)
    MIDI_file_output.addTempo(0, 0, 60)

    if H_normalization:
        H_max = np.amax(activation)
    else:
        H_max = 1

    for note_index in range(0, activation.shape[0]):  # Looping over the notes
        # Avoiding an uncontrolled situation (boolean to True before looking at this notes)
        if presence_of_a_note:
            presence_of_a_note = False

        for time_index in range(activation[note_index].size):  # Taking each time bin (discretized in 64ms windows)
            # Looking if the activation of this note over several consecutive frames is strong enough, being larger than a defined threshold
            # onsetCondition = (0.75 * activation[note_index, time_index] > Constants.NOTE_ACTIVATION_THRESHOLD) # Note detected
            minimalSustainCondition = (np.mean(
                activation[note_index, time_index:time_index + sliding_window]) > threshold * H_max)  # Note sustained

            if minimalSustainCondition:  # note detected and sustained
                if not presence_of_a_note:  # If the note hasn't been detected before
                    try:
                        current_pitch = midi_codebook[note_index]  # Storing the pitch of the actual note
                        for i in range(sliding_window):
                            onset_time_index = time_index + i
                            if (activation[note_index, onset_time_index] > pourcentage_onset_threshold * threshold):
                                current_onset = stft.time_bins[onset_time_index]  # Storing the onset
                                presence_of_a_note = True  # Note detected (for the future frames)
                                break
                    except ValueError as err:
                        # An error occured, the note is incorrect
                        print("The " + str(note_index) + " of the codebook is incorrect: " + err.args[1])
                        break

            else:
                if presence_of_a_note and stft.time_bins[time_index] > current_onset:  # End of the previous note
                    current_offset = stft.time_bins[time_index]
                    note_tab.append([current_onset, current_offset, current_pitch])  # Format for the .txt
                    MIDI_file_output.addNote(0, 0, current_pitch, current_onset, current_offset - current_onset,
                                             100)  # Adding in the .mid file

                    presence_of_a_note = False  # Reinitializing the detector of a note

    return note_tab, MIDI_file_output


def transcribe_activations_dynamic(midi_codebook, H, stft, threshold, sliding_window=10, H_normalization=False,
                                   minimum_note_duration_scale=1):
    presence_of_a_note = False
    note_tab = []
    current_pitch = 0
    current_onset = 0
    current_offset = 0

    # Creation of a .mid file
    MIDI_file_output = MIDIFile(1)
    MIDI_file_output.addTempo(0, 0, 60)

    H = np.zeros((88, 100))

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i][j] = (i * j) % 100

    # smoothing activation matrix (moving average)
    activation = np.zeros(np.shape(H))
    for i in range(0, np.shape(H)[1]):
        if i - sliding_window < 0 or i + sliding_window + 1 > np.shape(H)[1]:
            d = min(i, np.shape(H)[1] - i)
            activation[:, i] = np.mean(H[:, i - d:i + d + 1])
            f = 0
        else:
            activation[:, i] = np.mean(H[:, i - sliding_window:i + sliding_window + 1], axis=1)

    if H_normalization:
        H_max = np.amax(activation)
    else:
        H_max = 1

    for note_index in range(0, activation.shape[0]):  # Looping over the notes
        # Avoiding an uncontrolled situation (boolean to True before looking at this notes)
        if presence_of_a_note:
            presence_of_a_note = False

        for time_index in range(activation[note_index].size):
            # Looking if the activation of the note is larger than its smooth value + threshold
            minimalSustainCondition = (H[note_index, time_index] - activation[
                note_index, time_index] > threshold * H_max)  # actived note

            if minimalSustainCondition:  # note detected and sustained
                if not presence_of_a_note:  # If the note hasn't been detected before
                    try:
                        current_pitch = midi_codebook[note_index]  # Storing the pitch of the actual note
                        current_onset = stft.time_bins[time_index]  # find the onset time
                        presence_of_a_note = True  # Note detected (for the future frames)
                    except ValueError as err:
                        # An error occured, the note is incorrect
                        print("The " + str(note_index) + " of the codebook is incorrect: " + err.args[1])
                        break
            else:
                if presence_of_a_note:  # End of the previous note
                    current_offset = stft.time_bins[time_index]
                    note_tab.append([current_onset, current_offset, current_pitch])  # Format for the .txt
                    MIDI_file_output.addNote(0, 0, current_pitch, current_onset,
                                             (current_offset - current_onset) * minimum_note_duration_scale, 100)
                    presence_of_a_note = False  # Reinitializing the detector of a note

    return note_tab, MIDI_file_output


# Conversion of the fundamental frequencies in MIDI integer
# https://en.wikipedia.org/wiki/MIDI_tuning_standard#Frequency_values
def freq_to_midi(frequency):
    """
    Returns the frequency (Hz) in the MIDI scale

    Parameters
    ----------
    frequency: float
        Frequency in Hertz

    Returns
    -------
    midi_f0: integer
        Frequency in MIDI scale
    """
    return int(round(69 + 12 * math.log(frequency / 440, 2)))


def midi_to_freq(midi_freq):
    """
    Returns the MIDI frequency in Hertz

    Parameters
    ----------
    midi_freq: integer
        Frequency in MIDI scale

    Returns
    -------
    frequency: float
        Frequency in Hertz
    """
    return 440 * 2 ** ((midi_freq - 69) / 12)
