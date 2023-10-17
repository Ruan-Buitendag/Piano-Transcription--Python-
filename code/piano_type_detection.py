import math
import os
import re
import time
import STFT
import numpy as np
import activation_functions as scr

path_maps = "../MAPS"
path_templates = "../data_persisted/STFT/4096/tmp_W"

note_intensity = "M"


def CalculateTemplateWeights(midi_note, template_pianos, piano_H):
    piano_template_list = []

    for filename in os.listdir(path_templates):
        if filename.endswith('.npy'):  # Adjust the extension as needed
            parts = filename.split('_')

            if parts[0] != "W":
                continue

            if parts[4] in template_pianos and parts[-3] == midi_note:
                piano_template_list.append(np.load(os.path.join(path_templates, filename)))

    piano_templates = np.array(piano_template_list)

    piano_templates = np.swapaxes(piano_templates, 1, 2)
    piano_templates = np.swapaxes(piano_templates, 0, 2)
    piano_templates = np.squeeze(piano_templates, axis=3)

    path_this_song = "{}/{}/ISOL/NO/MAPS_ISOL_NO_M_S1_M{}_{}.wav".format(path_maps, piano_H, midi_note, piano_H)
    if not os.path.isfile(path_this_song):
        path_this_song = "{}/{}/ISOL/NO/MAPS_ISOL_NO_M_S0_M{}_{}.wav".format(path_maps, piano_H, midi_note, piano_H)

    H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf(path_this_song, 1, 50, 0.05, piano_templates,
                                                             time_limit=10,
                                                             H0=None, plot=False, channel="Sum",
                                                             num_bins=4096, skip_top=2000)

    means = np.mean(H, axis=1)
    means = means / np.sum(means)
    # print("Means are: ", means)

    return means


def CalculateTemplateWeights_Blind(spec, template_pianos):
    diffs = np.diff(spec, 1, axis=1)

    column_mask = (diffs > 0.2).any(axis=0)

    first_column_with_condition = np.argmax(column_mask)

    aaaaaaaaaaaaaa = []
    jjjjjjjjjjjj = []

    # for n in range(0, spec.shape[1] - 15, 5):
    # for n in range(21, 90):
    # for n in range(-5, 5):
    for n in range(0,1):

        piano_template_list = []

        start = first_column_with_condition
        end = start + 15

        note_spec = spec[:, start:end]

        bins_summed_over_time = np.mean(note_spec, axis=1)

        fundamental_bin = np.argmax(bins_summed_over_time)

        fund_frequency = fundamental_bin / 4096 * 22050

        if fund_frequency <= 0:
            continue

        midi_note = str(hertz_to_midi(fund_frequency) + n)
        # midi_note = str(n)

        for filename in os.listdir(path_templates):
            if filename.endswith('.npy'):  # Adjust the extension as needed
                parts = filename.split('_')

                if parts[0] != "W":
                    continue

                if parts[4] in template_pianos and parts[-1].split('.')[0] == midi_note:
                    piano_template_list.append(np.load(os.path.join(path_templates, filename)))

        piano_templates = np.array(piano_template_list)

        if piano_templates.shape[0] == 0:
            continue

        piano_templates = np.swapaxes(piano_templates, 1, 2)
        piano_templates = np.swapaxes(piano_templates, 0, 2)
        piano_templates = np.squeeze(piano_templates, axis=3)

        H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf_from_spec(note_spec, 1, 5, 0.05, piano_templates,
                                                                           time_limit=10,
                                                                           H0=None, plot=False, channel="Sum",
                                                                           num_bins=4096, skip_top=1000)


        means = np.mean(H, axis=1)
        means = means / np.sum(means)
        # print("Means are: ", means)

        means = np.append(means, all_err[-1])

        aaaaaaaaaaaaaa.append(means)
        jjjjjjjjjjjj.append([means, start, end, midi_note, bin_from_midi(int(midi_note))])

    aaa = np.array(aaaaaaaaaaaaaa)

    # bb = np.argmax(aaa)
    # cc = np.unravel_index(bb, aaa.shape)
    # means = aaa[cc[0], :]

    row = np.argmin(aaa[:, -1])
    means = aaa[row, :-1]

    a = 7

    return means


def hertz_to_midi(frequency):
    midi_note = 12 * math.log2(frequency / 440) + 69
    return round(midi_note)


def midi_to_hertz(midi_note):
    frequency = 440 * math.pow(2, (midi_note - 69) / 12)
    return frequency


def bin_from_midi(midi_note):
    frequency = midi_to_hertz(midi_note)
    bin = frequency / 22050 * 4096
    return bin


stft = STFT.STFT("C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.wav", 10, "Average")

spec = stft.get_magnitude_spectrogram()

print(CalculateTemplateWeights_Blind(spec, ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkCl", "SptkBGAm", "SptkBGCl", "StbgTGd2"]))
