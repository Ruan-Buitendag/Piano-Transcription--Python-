import numpy as np
import os
import transcription_functions as scr
import sys
import time
import STFT
import evaluate_transcription as et
import mir_eval
import transcribe_factorization as tf

if __name__ == "__main__":
    # Parameters and paths
    pianos = ["AkPnCGdD", "ENSTDkCl", "AkPnBcht", "AkPnBsdf", "AkPnStgb", "ENSTDkAm", "SptkBGAm", "StbgTGd2"]

    path_root_maps = "../MAPS"

    piano_W = "AkPnBcht"
    piano_H = "AkPnBcht"
    stft_bins = 4096
    # song = "MAPS_MUS-bach_847_" + piano_H + ".wav"
    song = "MAPS_MUS-chpn_op66_AkPnBcht.wav"



    note_intensity = "M"
    beta = 1

    T_array = [10]

    print(f"Piano templates learned on: {piano_W}")

    path_songs = "{}/{}/MUS".format(path_root_maps, piano_H)
    persisted_path = "../data_persisted/STFT/" + str(stft_bins)

    itmax_W = 500
    init = "L1"

    time_limit = 30
    itmax_H = 20
    tol = 1e-8

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
        W_persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_{}".format(piano_W,
                                                                                                            beta, T,
                                                                                                            init,
                                                                                                            True,
                                                                                                            itmax_W,
                                                                                                            note_intensity)
        try:
            dict_W = np.load("{}/{}.npy".format(persisted_path, W_persisted_name))
        except FileNotFoundError:
            raise FileNotFoundError("Dictionary could not be found, to debug (probably a wrong T)")

        song_name = song.replace(".wav", "")
        print("processing piano song: {}".format(song_name))
        path_this_song = "{}/{}".format(path_songs, song)
        H_to_persist_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(
            song_name, piano_W, beta, T, init, True, itmax_H, note_intensity, time_limit, tol)

        try:
            np.load("aaaaa.npy")
            np.load("{}/activations/{}.npy".format(persisted_path, H_to_persist_name), allow_pickle=True)
            print("Found in loads.")
        except FileNotFoundError:
            time_start = time.time()



            H, n_iter, all_err = scr.semi_supervised_transcribe_cnmf(path_this_song, beta, itmax_H, tol, dict_W,
                                                                     time_limit=time_limit,
                                                                     H0=None, plot=False, model_AD=True,
                                                                     channel="Sum", num_bins=stft_bins)
            print("Time: {}".format(time.time() - time_start))

            np.save("{}/activations/{}".format(persisted_path, H_to_persist_name), H)

        print("Done determining activation matrix.")
        print("Post processing activations.")

        stft = STFT.STFT(path_this_song, time=time_limit, channel=0, num_bins=stft_bins)

        annot_name = song.replace("wav", "txt")
        annot_this_song = "{}/{}".format(path_songs, annot_name)
        note_annotations = et.load_ref_in_array(annot_this_song, time_limit=time_limit)
        ref = np.array(note_annotations, float)
        ref_pitches = np.array(ref[:, 2], int)

        res_a_param = []
        H_persisted_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(
            song_name, piano_W, beta, T, init, True, itmax_H, note_intensity, time_limit, tol)
        H = np.load("{}/{}.npy".format("../data_persisted/STFT/" + str(stft_bins) + "/activations", H_persisted_name),
                    allow_pickle=True)
        all_res = []

        res_every_thresh = []
        for threshold in listthres:
            prediction, midi_file_output = tf.transcribe_activations_dynamic(codebook, H, stft, threshold,
                                                                             H_normalization=False,
                                                                             minimum_note_duration_scale=40)

            est = np.array(prediction, float)

            output_file_path = str(threshold) + '.mid'

            # Save the MIDIFile object to the specified file path
            with open(output_file_path, 'wb') as output_file:
                midi_file_output.writeFile(output_file)

            print(f"MIDI file saved to {output_file_path}")

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
