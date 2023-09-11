import numpy as np
import mir_eval.transcription
import os
from IPython.display import display, Markdown
from midiutil import MIDIFile

import transcribe_factorization as tf
import evaluate_transcription as et
import STFT
import pandas as pd

import note_seq as ns



def printmd(string):
    display(Markdown(string))

def load_ref_in_array_midi(ref_path, time_limit=None):

    truth_array = []
    note_seq = ns.midi_file_to_note_sequence(ref_path).notes
    for i in range(len(note_seq)):
        note = note_seq[i]
        line_to_array = [note.start_time, note.end_time, note.pitch]

        if(time_limit != None) and (float(line_to_array[0]) > time_limit):
            # if onset > time_limit (note outside of the cropped excerpt)
            break
        else:
            truth_array.append(line_to_array)

    return truth_array

def compute_scores_database(piano_type_W, piano_type_H, H_normalization = False, adaptative_threshold = True, path_computed_H = "../data_persisted/activations", path_MAPS = "../MAPS"):
    path_songs = path_MAPS+"/{}/MUS".format(piano_type_H)
    print("## Piano for W: {}, and for H: {}".format(piano_type_W, piano_type_H))

    time_limit = 30
    beta = 1
    init = "L1"
    model_AD = True
    note_intensity = "M"
    itmax_H = 50
    tol = 1e-8
    codebook = range(21, 109)
    onset_tolerance = 50/1000

    f = np.arange(1e-2, 4e-1, 2e-2)
    listthres = np.r_[f[::-1]]

    files = os.listdir(path_songs)
    list_files_wav = []
    for it_files in files:
        if it_files.split(".")[-1] == "wav":
            list_files_wav.append(it_files)

    all_res = []
    for a_song in list_files_wav:
        song_name = a_song.replace(".wav", "")
        print("processing piano song: {}".format(song_name))
        path_this_song = "{}/{}".format(path_songs, a_song)
        stft = STFT.STFT(path_this_song, time = time_limit, channel = 0)

        #X = stft.get_magnitude_spectrogram()

        annot_name = a_song.replace("wav","txt")
        annot_this_song = "{}/{}".format(path_songs, annot_name)
        note_annotations = et.load_ref_in_array(annot_this_song, time_limit=time_limit)
        ref = np.array(note_annotations, float)
        ref_pitches = np.array(ref[:,2], int)
        try:
            #res_each_song = []
            res_a_param = []
            for T in [10]:
                H_persisted_name = "activations_song_{}_W_learned_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_W_{}_time_limit_{}_tol_{}".format(song_name, piano_type_W, beta, T, init, model_AD, itmax_H, note_intensity, time_limit, tol)
                H = np.load("{}/{}.npy".format(path_computed_H, H_persisted_name), allow_pickle = True)
                res_every_thresh = []
                for threshold in listthres:
                    if adaptative_threshold:
                        prediction, _ = tf.transcribe_activations_dynamic(codebook, H, stft, threshold, H_normalization = H_normalization)
                    else:
                        prediction, _ = tf.transcribe_activations(codebook, H, stft, threshold, H_normalization = H_normalization)
                    est = np.array(prediction, float)
                    if est.size > 0:
                        est_pitches = np.array(est[:,2], int)
                        (prec, rec, f_mes, _) = mir_eval.transcription.precision_recall_f1_overlap(ref[:,0:2], ref_pitches, est[:,0:2], est_pitches, offset_ratio = None, onset_tolerance = onset_tolerance)
                        matching = mir_eval.transcription.match_notes(ref[:,0:2], ref_pitches, est[:,0:2],est_pitches, onset_tolerance=onset_tolerance,offset_ratio=None)
                        TP = len(matching)
                        try:
                            FP = int(TP * (1 - prec) / prec)
                        except ZeroDivisionError:
                            FP = 0
                        try:
                            FN = int(TP * (1 - rec) / rec)
                        except ZeroDivisionError:
                            FN = 0
                        acc = et.accuracy(TP,FP,FN)
                    else:
                        prec, rec, f_mes, acc, TP, FP, FN = (0,0,0,0,0,0,0)
                    res_every_thresh.append([prec, rec, f_mes, acc, TP, FP, FN])
                res_a_param.append(res_every_thresh)
            #res_each_song.append(res_a_param)

            all_res.append(res_a_param)

        except FileNotFoundError:
            print("\033[91m This song failed: {} \033[00m".format(a_song))
            pass
    np_all_res = np.array(all_res)
    the_t = []
    # for t in [10]:
    for t in [10]:
        the_t.append("T: {}".format(t))
    index_pandas = the_t
    col = ['Best threshold','Precision', 'Recall', 'F measure','Accuracy','True Positives','False Positives','False Negatives']
    lines = []
    lines_opt_thresh = []
    for cond in range(len(index_pandas)):
        all_thresh = []
        for each_thresh in range(len(listthres)):
            all_thresh.append(np.mean(np_all_res[:,cond,each_thresh,2]))
        best_thresh_idx = np.argmax(all_thresh)
        this_line = [listthres[best_thresh_idx]]
        for i in range(len(col) - 1):# - 1 because threshold
            this_line.append(round(np.mean(np_all_res[:,cond,best_thresh_idx,i]), 4))
        lines.append(this_line)

        best_val = []
        for each_song in range(len(list_files_wav)):
            best_thresh_idx = np.argmax(np_all_res[each_song,cond,:,2])
            best_val.append([round(np_all_res[each_song,cond,best_thresh_idx,i], 4) for i in range(len(col) - 1)])
        lines_opt_thresh.append([round(np.mean(np.array(best_val)[:,i]),4) for i in range(len(col) - 1)])

    printmd("### When averaging each threshold on all MAPS")
    df = pd.DataFrame(np.array(lines), columns = col, index = index_pandas)
    display(df.style.bar(subset=["F measure", "Accuracy"], color='#5fba7d'))

    printmd("### When optimizing the threshold on each song")
    best_df = pd.DataFrame(np.array(lines_opt_thresh), columns = col[1:], index = index_pandas)
    display(best_df.style.bar(subset=["F measure", "Accuracy"], color='#5fba7d'))
