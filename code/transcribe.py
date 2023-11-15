import csv

import GPy

import numpy as np
import os
import activation_functions as scr
import time
import STFT
import evaluate_transcription as et
import mir_eval
import transcribe_factorization as tf
import subprocess
import convolutive_MM as MM

#  y = 1.8223x + 0.6118


import myfmeasure

import domain_adaptation as da

import single_note_eq_mask as eq
import piano_type_detection as pt

if __name__ == "__main__":
    # Specify the file path of the CSV file to delete
    csv_file_path = 'treshold_data.csv'

    # Check if the file exists before attempting to delete it
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f'The file {csv_file_path} has been deleted.')
    else:
        print(f'The file {csv_file_path} does not exist.')

    with(open('treshold_data.csv', 'a', newline='')) as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header to the CSV file
        csv_writer.writerow(
            ['Threshold', 'Mean', 'Max', 'Median', 'Variance', 'Number above mean', 'Number above median'])

    # threshold_model = GPy.models.GPRegression.load_model("testmodel.zip")

    # Parameters and paths
    pianos = ["AkPnCGdD", "ENSTDkCl", "AkPnBcht", "AkPnBsdf", "AkPnStgb", "ENSTDkAm", "SptkBGAm", "StbgTGd2"]

    path_root_maps = "../MAPS"
    path_fluidsynth_exe = "C:/tools/fluidsynth/bin/fluidsynth.exe"
    path_soundfont = "../soundfonts/yamaha_piano.sf2"

    piano_W = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkAm", "ENSTDkCl", "SptkBGAm", "StbgTGd2"]
    #
    piano_W = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkAm"]
    #
    # piano_W = ["AkPnBcht"]

    # piano_W = "AkPnBcht"
    # piano_H = "AkPnBcht"
    pianos_H = ["AkPnBcht"]

    # pianos_H = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkAm", "ENSTDkCl", "SptkBGAm", "StbgTGd2"]

    midi_note_for_eq = "59"
    every_note = False

    spec_type = "stft"
    num_points = 4096

    note_length = 5

    specific_song = None
    # specific_song = "MAPS_MUS-chpn_op27_2_SptkBGCl.wav"
    # specific_song = "MAPS_MUS-bk_xmas2_SptkBGCl.wav"
    # specific_song = "MAPS_MUS-chpn-p4_AkPnBcht.wav"
    # specific_song = "MAPS_MUS-chpn_op66_AkPnBcht_NP.wav"
    # specific_song = "MAPS_MUS-alb_esp2_AkPnCGdD.wav"
    specific_song = "MAPS_MUS-alb_se3_AkPnBcht-1.wav"
    # specific_song = "MAPS_MUS-alb_se3_AkPnBsdf.wav"
    # specific_song =  "Freeze Noise.wav"
    # specific_song = "aaaaaaaaaaaaaaaaaaa.wav"

    if specific_song is not None:
        piano_H = specific_song.split("_")[-1].split(".")[0]

    skip_top = 4096 - 1499
    # skip_top = 0

    time_limit = 5
    itmax_H = 100

    re_activate = True
    # re_activate = False

    note_intensity = "M"
    beta = 1

    T_array = [10]

    for piano_H in pianos_H:

        print(f"Piano templates learned on: {piano_W}")

        path_songs = "{}/{}/LIVE_MUS".format(path_root_maps, piano_H)

        if spec_type == "stft":
            persisted_path = "../data_persisted/STFT/" + str(num_points)
        elif spec_type == "mspec":
            persisted_path = "../data_persisted/MSPEC/" + str(num_points)

        itmax_W = 500
        init = "L1"

        tol = 1e-8

        f = np.arange(0, 0.5, 0.001)
        listthres = np.r_[f[::-1]]
        # listthres = [0.1]
        codebook = range(21, 109)

        onset_tolerance = 0.05

        files = os.listdir(path_songs)
        list_files_wav = []
        for it_files in files:
            if it_files.split(".")[-1] == "wav":
                list_files_wav.append(it_files)

        if specific_song is not None:
            list_files_wav = [specific_song]

        all_song_thresholds = []
        all_song_f_scores = []

        for song in list_files_wav:

            if "-1" not in song:
                continue

            song_name = song.replace(".wav", "")
            print("processing piano song: {}".format(song_name))
            path_this_song = "{}/{}".format(path_songs, song)

            if type(piano_W) == str:
                W_persisted_name = "conv_dict_piano_{}".format(piano_W)
                try:
                    dict_W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))

                    if dict_W.shape[0] != 10:
                        raise ValueError("Dictionary has the incorrect number of convolutional kernels")
                    if dict_W.shape[1] != num_points + 1:
                        raise ValueError("Dictionary has the incorrect number of frequency bins")

                    # dict_W = da.EQDictionary(dict_W, piano_H, piano_W)
                    # dict_W = da.EQDictionaryFromSingleNote(dict_W, piano_H, piano_W, midi_note_for_eq)

                except FileNotFoundError:
                    raise FileNotFoundError("Dictionary could not be found")

            else:
                list_dicts_W = []

                for piano_W_it in piano_W:
                    W_persisted_name = "conv_dict_piano_{}".format(
                        piano_W_it)
                    try:
                        list_dicts_W += [np.load("{}/{}.npy".format(persisted_path, W_persisted_name))]
                    except FileNotFoundError:
                        print(piano_W_it, " does not have a corresponding dictionary")

                dict_W = np.zeros(list_dicts_W[0].shape)
                dicts_W = np.array(list_dicts_W)

                stft = STFT.STFT(path_this_song, time=time_limit, channel=0, num_bins=num_points)
                aaa = stft.get_magnitude_spectrogram()

                try:
                    weights = pt.CalculateTemplateWeights_Blind(aaa, piano_W)
                    # weights = pt.CalculateTemplateWeights(midi_note_for_eq, piano_W, piano_H)
                    weights = np.pad(weights, (0, dicts_W.shape[0] - weights.shape[0]), 'constant', constant_values=0)

                    # weights = pt.CalculateTemplateWeights_Blind_with_time(aaa, piano_W)

                    # weights = np.pad(weights, (0, dicts_W.shape[0] - weights.shape[0]), 'constant', constant_values=0)

                except:
                    print("Could not calculate weights for ", song_name)
                    continue

                print("weights: ", weights)

                weights = weights[:, np.newaxis, np.newaxis]
                # weights = weights[:, :, np.newaxis]

                for note in range(88):
                    if every_note:
                        weights = np.array(pt.CalculateTemplateWeights(str(note + 21), piano_W, piano_H))

                        weights = np.pad(weights, (0, dicts_W.shape[0] - weights.shape[0]), 'constant',
                                         constant_values=0)

                        weights = weights[:, np.newaxis, np.newaxis]
                        # weights = weights[:, :, np.newaxis]

                    dict_W[:, :, note] = np.sum(weights * dicts_W[:, :, :, note], axis=0)

                # dict_W = da.EQDictionary(dict_W, piano_H, piano_W)
                # dict_W = da.EQDictionaryFromSingleNote(dict_W, piano_H, piano_W, midi_note_for_eq)

            H_to_persist_name = "activations_song_{}_W_learned_{}_itmax_{}_timelimit_{}".format(
                song_name, piano_W, itmax_H, time_limit)

            try:
                if re_activate:
                    np.load("aaaaa.npy")
                np.load("{}/activations/{}.npy".format(persisted_path, H_to_persist_name), allow_pickle=True)
                print("Found in loads.")
            except FileNotFoundError:
                time_start = time.time()

                H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf(path_this_song, beta, itmax_H, tol, dict_W,
                                                                         time_limit=time_limit,
                                                                         H0=None, plot=False, channel="Sum",
                                                                         num_bins=num_points, skip_top=skip_top)
                print("Time: {}".format(time.time() - time_start))

                np.save("{}/activations/{}".format(persisted_path, H_to_persist_name), H)

            except OSError:
                print("{}/activations/{}.npy".format(persisted_path, H_to_persist_name), " filename too long")
                continue

            print("Done determining activation matrix.")
            print("Post processing activations.")

            stft = STFT.STFT(path_this_song, time=time_limit, channel=0, num_bins=num_points)
            aaa = stft.get_magnitude_spectrogram()

            delay = stft.getDelay()

            annot_name = song.replace("-1.wav", ".txt")
            # annot_name = "MAPS_MUS-alb_se3_AkPnBcht.txt"
            annot_this_song = "{}/{}".format(path_songs, annot_name)
            note_annotations = et.load_ref_in_array(annot_this_song, time_limit=time_limit - delay)
            # note_annotations = et.load_ref_in_array(annot_this_song, time_limit=time_limit)
            ref = np.array(note_annotations, float)
            ref_pitches = np.array(ref[:, 2], int)

            res_a_param = []
            H_persisted_name = "activations_song_{}_W_learned_{}_itmax_{}_timelimit_{}".format(
                song_name, piano_W, itmax_H, time_limit)

            if spec_type == "stft":
                H_directory = "../data_persisted/STFT/"
            elif spec_type == "mspec":
                H_directory = "../data_persisted/MSPEC/"

            H = np.load("{}/{}.npy".format(H_directory + str(num_points) + "/activations", H_persisted_name),
                        allow_pickle=True)

            # above_thresh =
            H_aaa = H[H > 0.01]

            H_mean = np.mean(H_aaa)
            H_max = np.max(H_aaa)
            H_median = np.median(H_aaa)
            H_var = np.var(H_aaa)
            H_num_above_mean = np.sum(H_aaa > H_mean)
            H_num_above_median = np.sum(H_aaa > H_median)

            all_res = []

            res_every_thresh = []

            # listthres = [9.8125 * H_var + 0.0318]
            # listthres = [3.7774 * H_median - 0.0308 -0.02]
            # listthres = [0.043]
            #
            # idddd = np.zeros((1, 2))
            # idddd[:, 0] = H_var
            # idddd[:, 1] = H_mean
            #
            # ddd = threshold_model.predict(idddd)[0][0][0]
            #
            # listthres = [ddd]

            for threshold in listthres:
                prediction, midi_file_output = tf.transcribe_activations_dynamic(codebook, H, stft, threshold,
                                                                                 H_normalization=False,
                                                                                 minimum_note_duration_scale=note_length)

                est = np.array(prediction, float)

                output_file_path = "../transcriptions/" + str(round(threshold, 2)) + '.mid'

                # Save the MIDIFile object to the specified file path
                with open(output_file_path, 'wb') as output_file:
                    midi_file_output.writeFile(output_file)

                # print(f"MIDI file saved to {output_file_path}")

                if est.size > 0:
                    bu_est = np.copy(est)

                    est[:, :2] = est[:, :2] - np.min(est[:, 0])
                    ref[:, :2] = ref[:, :2] - np.min(ref[:, 0])

                    est_pitches = np.array(est[:, 2], int)
                    # (prec, rec, f_mes, _) = mir_eval.transcription.precision_recall_f1_overlap(ref[:, 0:2], ref_pitches,
                    #                                                                            est[:, 0:2], est_pitches,
                    #                                                                            offset_ratio=None,
                    #                                                                            onset_tolerance=onset_tolerance)

                    (prec, rec, f_mes, TP) = myfmeasure.stats(ref[:, 0:2], ref_pitches, est_pitches, est[:, 0:2])

                    # matching = mir_eval.transcription.match_notes(ref, ref_pitches, est, est_pitches,
                    #                                               onset_tolerance=onset_tolerance, offset_ratio=None)
                    # TP = len(matching)
                    try:
                        FP = round(TP * (1 - prec) / prec)
                    except ZeroDivisionError:
                        FP = 0
                    try:
                        FN = round(TP * (1 - rec) / rec)
                    except ZeroDivisionError:
                        FN = 0
                    acc = et.accuracy(TP, FP, FN)
                else:
                    prec, rec, f_mes, acc, TP, FP, FN = (0, 0, 0, 0, 0, 0, 0)
                res_every_thresh.append([prec, rec, f_mes, acc, TP, FP, FN])
            res_a_param.append(res_every_thresh)

            all_res.append(res_a_param)

            threshold_results = np.array(all_res[0][0])

            f_score_max_index = threshold_results[:, 2].argmax()
            best_thresh = listthres[f_score_max_index]

            best_results = threshold_results[f_score_max_index, :]

            print("Best threshold: ", best_thresh)
            print("F-score: ", best_results[2])
            print("Accuracy: ", best_results[3])
            print("Precision: ", best_results[0])
            print("Recall: ", best_results[1])
            print("TP: ", best_results[4])
            print("FP: ", best_results[5])
            print("FN: ", best_results[6])

            all_song_thresholds += [best_thresh]
            all_song_f_scores += [best_results[2]]

            with open('treshold_data.csv', 'a', newline='') as csv_file:
                # Create a CSV writer object
                csv_writer = csv.writer(csv_file)

                # Write the data to the CSV file
                csv_writer.writerow([best_thresh, H_mean, H_max, H_median, H_var, H_num_above_mean, H_num_above_median])

            # Define the paths you want to pass as parameters
            path_to_best_song = "../transcriptions/" + str(round(listthres[f_score_max_index], 2)) + '.mid'

            # Construct the command to run the executable with the paths as parameters
            # command = [path_fluidsynth_exe, path_soundfont, path_to_best_song]
            #
            # if specific_song is not None:
            #     try:
            #         # Start the process and connect to its stdin for sending input
            #         process = subprocess.run(command)
            #
            #     except subprocess.CalledProcessError as e:
            #         print("An error occurred:", e)

        print("All song thresholds: ", all_song_thresholds)
        print("Mean thresh: ", np.mean(all_song_thresholds))
        print("All song f-scores: ", all_song_f_scores)
        print("Mean f-score: ", np.mean(all_song_f_scores))
