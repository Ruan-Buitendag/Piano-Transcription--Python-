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
