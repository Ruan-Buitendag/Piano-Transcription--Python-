import numpy as np
import os
import re
import time
import convolutive_MM as MM
from numba import jit
import STFT

persisted_path = "../data_persisted/STFT/2048"

def learning_W_and_persist(path, beta, T, itmax=500, rank=1, init="random", model_AD = True, piano_type = "ENSTDkCl", note_intensity = "F"):
    """
    Learning of note template using isolated note recording in MAPS
    ---------------
    :param path: isolated notes path
    :param beta: coefficient beta
    :param T: number of convolutive dictionary
    :param itmax: maximal iteration number
    :param rank: factorization rank
    :return: A note dictionary
    """
    files = os.listdir(path)
    list_files_wav = []
    for it_files in files:
        if it_files.split(".")[-1] == "wav" and it_files.split("_")[-4] == note_intensity:
            list_files_wav.append(it_files)

    if len(list_files_wav) == 0:
        raise NotImplementedError("Empty list of songs.")
    Dictionary_MM = {}
    Dictionary_H = {}

    for name in list_files_wav:
        f = path + "/" + name
        midi = re.search(r'(?<=M)\d+', name).group(0)
        print('MIDI: ', midi)
        try:
            persisted_name = "W_one_note_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_midi_{}_intensity_{}".format(piano_type, beta, T, init, model_AD, itmax, midi, note_intensity)
            W_mm = np.load("{}/tmp_W/{}.npy".format(persisted_path, persisted_name), allow_pickle = True)
            persisted_name = "H" + persisted_name[1:]
            H = np.load("{}/tmp_W/{}.npy".format(persisted_path, persisted_name), allow_pickle = True)
            print("Found in loads")
        except FileNotFoundError:
            time_start = time.time()
            stft = STFT.STFT(f, model_AD=model_AD)
            mag = stft.get_magnitude_spectrogram()

            # we remove the column if all elements in that column < 1e-10
            columnlist = []
            for i in range(np.shape(mag)[1]):
                if (mag[:, i] < 1e-10).all():
                    columnlist.append(i)
            mag = np.delete(mag, columnlist, axis=1)

            if init == "L1":
                W0, H0 = L1_initialization(mag, T)

            else:
                raise NotImplementedError("Wrong init parameter: {}".format(init))

            [W_mm, H,_,all_err] = MM.convlutive_MM(mag, rank, itmax, beta, T, 1e-7, W0=W0, H0=H0)

            #Persist W_mm
            persisted_name = "W_one_note_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_midi_{}_intensity_{}".format(piano_type, beta, T, init, model_AD, itmax, midi, note_intensity)
            np.save("{}/tmp_W/{}".format(persisted_path, persisted_name), W_mm)

            H_persisted_name = "H" + persisted_name[1:]
            np.save("{}/tmp_W/{}".format(persisted_path, H_persisted_name), H)
            print("time:{}".format(time.time() - time_start))

        Dictionary_MM[int(midi)] = W_mm
        Dictionary_H[int(midi)] = max(H.flatten())

    # build dictionary
    mat_mm = np.array([np.zeros(shape=(np.shape(W_mm)[1], 88))]*T)
    max_value_h = np.zeros(88)
    for t in range(T):
        for i in range(88):
            mat_mm[t][:, i] = Dictionary_MM[i + 21][t].flatten() # Flatten?
    for i in range(88):
        max_value_h[i] = Dictionary_H[i + 21]

    persisted_name = "conv_dict_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_{}".format(piano_type, beta, T, init, model_AD, itmax, note_intensity)
    np.save("{}/{}".format(persisted_path, persisted_name), mat_mm)

    h_persisted_name = "max_value_h_piano_{}_beta_{}_T_{}_init_{}_stftAD_{}_itmax_{}_intensity_{}".format(piano_type, beta, T, init, model_AD, itmax, note_intensity)
    np.save("{}/{}".format(persisted_path, h_persisted_name), max_value_h)

    return mat_mm, max_value_h

def L1_initialization(mag, T):
    # find the W with largest norm L1
    ncol = np.shape(mag)[1]
    list_norm = [np.linalg.norm(mag[:, j], ord=1) for j in range(ncol)]
    index = -1
    obj = -10
    for i in range(ncol - T):
        m = np.mean(list_norm[i:i + T])
        if m > obj:
            index = i
            obj = m
    # print(W)
    W = mag[:, index:index + T]
    # compute H using gradient descent
    H0 = np.array([[1e-10] * ncol])
    # H0 = np.zeros(shape= (1, ncol))
    for i in range(T):
        H0[:,index+T] = 1

    W1 = np.transpose(W)
    W0 = np.transpose(W)[:, :, np.newaxis]

    return W0, H0
