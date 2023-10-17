import h5py
import numpy as np

import STFT

import numpy

# aaa = STFT.STFT("MAPS_MUS-alb_se3_AkPnBcht.wav", 10, "Average")
#
# aa = aaa.getDelay()
#
# aaa = aaa.get_magnitude_spectrogram()[:, round(aa / 0.02):]
#
# bbb = STFT.STFT("MAPS_MUS-alb_se3_AkPnBsdf.wav", 10, "Average")
#
# bb = bbb.getDelay()
#
# bbb = bbb.get_magnitude_spectrogram()[:, round(bb / 0.02):]
#
# ccc = numpy.mean(bbb, axis=1) - numpy.mean(aaa, axis=1)
#
# np.save("penis", ccc)
#
# a = 0

fff = np.load("C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/data_persisted/STFT/4096/tmp_W/W_one_note_piano_ENSTDkCl_midi_69.npy")

# read from hdf5 file

filename = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/single_notes/W_one_note_piano_ENSTDkCl_midi_69.h5"

with h5py.File(filename, 'r') as f:
    # List all the keys (groups/datasets) in the HDF5 file
    print("Keys:", list(f.keys()))

    # Access a dataset or group
    dataset = f['dictionary']  # Replace 'dataset_name' with the name of your dataset

    # Read data from the dataset
    data = dataset[()]  # This loads the entire dataset into a NumPy array

    a = 0