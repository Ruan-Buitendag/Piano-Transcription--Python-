import numpy as np
import os
import activation_functions as scr
import time
import STFT
import evaluate_transcription as et
import mir_eval
import transcribe_factorization as tf
import subprocess

if __name__ == "__main__":
    # Parameters and paths
    pianos = ["AkPnCGdD", "ENSTDkCl", "AkPnBcht", "AkPnBsdf", "AkPnStgb", "ENSTDkAm", "SptkBGAm", "StbgTGd2"]

    path_root_maps = "../MAPS"
    path_fluidsynth_exe = "C:/tools/fluidsynth/bin/fluidsynth.exe"
    path_soundfont = "../soundfonts/yamaha_piano.sf2"

    piano_W = "AkPnBcht"
    piano_H = "AkPnBcht"

    spec_type = "stft"
    num_points = 4096

    # song = "MAPS_MUS-bach_847_" + piano_H + ".wav"
    song = "MAPS_MUS-chpn_op66_AkPnBcht.wav"
    note_intensity = "M"
    beta = 1

    T_array = [10]

    print(f"Piano templates learned on: {piano_W}")

    path_songs = "{}/{}/MUS".format(path_root_maps, piano_H)

    if spec_type == "stft":
        persisted_path = "../data_persisted/STFT/" + str(num_points)
    elif spec_type == "mspec":
        persisted_path = "../data_persisted/MSPEC/" + str(num_points)

    itmax_W = 500
    init = "L1"

    time_limit = 30
    itmax_H = 20
    tol = 1e-8

    skip_top = 3500

    f = np.arange(1e-2, 4e-1, 1e-2)
    listthres = np.r_[f[::-1]]
    codebook = range(21, 109)

    onset_tolerance = 0.05

    files = os.listdir(path_songs)
    list_files_wav = []
    for it_files in files:
        if it_files.split(".")[-1] == "wav":
            list_files_wav.append(it_files)

    for T in T_array:
        print(f"T: {T}")
        W_persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_{}_{}_itmax_{}_intensity_{}".format(piano_W,
                                                                                                        beta, T,
                                                                                                        init, spec_type,
                                                                                                        num_points,
                                                                                                        itmax_W,
                                                                                                        note_intensity)
        try:
            dict_W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))

            if dict_W.shape[0] != T:
                raise ValueError("Dictionary has the incorrect number of convolutional kernels")
            if dict_W.shape[1] != num_points + 1:
                raise ValueError("Dictionary has the incorrect number of frequency bins")

        except FileNotFoundError:
            raise FileNotFoundError("Dictionary could not be found")

        song_name = song.replace(".wav", "")
        print("processing piano song: {}".format(song_name))
        path_this_song = "{}/{}".format(path_songs, song)
        H_to_persist_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_{}_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(
            song_name, piano_W, beta, T, init, spec_type, num_points, itmax_H, note_intensity, time_limit, tol)

        try:
            # np.load("aaaaa.npy")
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

        print("Done determining activation matrix.")
        print("Post processing activations.")

        stft = STFT.STFT(path_this_song, time=time_limit, channel=0, num_bins=num_points)

        annot_name = song.replace("wav", "txt")
        annot_this_song = "{}/{}".format(path_songs, annot_name)
        note_annotations = et.load_ref_in_array(annot_this_song, time_limit=time_limit)
        ref = np.array(note_annotations, float)
        ref_pitches = np.array(ref[:, 2], int)

        res_a_param = []
        H_persisted_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_{}_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(
            song_name, piano_W, beta, T, init, spec_type, num_points, itmax_H, note_intensity, time_limit, tol)

        if spec_type == "stft":
            H_directory = "../data_persisted/STFT/"
        elif spec_type == "mspec":
            H_directory = "../data_persisted/MSPEC/"

        H = np.load("{}/{}.npy".format(H_directory + str(num_points) + "/activations", H_persisted_name),
                    allow_pickle=True)

        all_res = []

        res_every_thresh = []
        for threshold in listthres:
            prediction, midi_file_output = tf.transcribe_activations_dynamic(codebook, H, stft, threshold,
                                                                             H_normalization=False,
                                                                             minimum_note_duration_scale=10)

            est = np.array(prediction, float)

            output_file_path = "../transcriptions/" + str(round(threshold, 2)) + '.mid'

            # Save the MIDIFile object to the specified file path
            with open(output_file_path, 'wb') as output_file:
                midi_file_output.writeFile(output_file)

            # print(f"MIDI file saved to {output_file_path}")

            if est.size > 0:
                est_pitches = np.array(est[:, 2], int)
                (prec, rec, f_mes, _) = mir_eval.transcription.precision_recall_f1_overlap(ref[:, 0:2], ref_pitches,
                                                                                           est[:, 0:2], est_pitches,
                                                                                           offset_ratio=None,
                                                                                           onset_tolerance=onset_tolerance)
                matching = mir_eval.transcription.match_notes(ref[:, 0:2], ref_pitches, est[:, 0:2], est_pitches,
                                                              onset_tolerance=onset_tolerance, offset_ratio=None)
                TP = len(matching)
                try:
                    FP = int(TP * (1 - prec) / prec)
                except ZeroDivisionError:
                    FP = 0
                try:
                    FN = int(TP * (1 - rec) / rec)
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

        best_results = threshold_results[f_score_max_index, :]

        print("Best threshold: ", listthres[f_score_max_index])
        print("F-score: ", best_results[2])
        print("Accuracy: ", best_results[3])
        print("TP: ", best_results[4])
        print("FP: ", best_results[5])
        print("FN: ", best_results[6])

        # Define the paths you want to pass as parameters
        path_to_best_song = "../transcriptions/" + str(round(listthres[f_score_max_index], 2)) + '.mid'

        # Construct the command to run the executable with the paths as parameters
        command = [path_fluidsynth_exe, path_soundfont, path_to_best_song]

        try:
            # Start the process and connect to its stdin for sending input
            process = subprocess.run(command)

        except subprocess.CalledProcessError as e:
            print("An error occurred:", e)
