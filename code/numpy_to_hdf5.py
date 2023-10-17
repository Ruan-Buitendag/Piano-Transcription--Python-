import os

import h5py
import numpy as np

def rename_dicts():
    dir = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/data_persisted/STFT/4096/"
    outputdir = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/dictionaries/"

    contents = os.listdir(dir)

    for filename in contents:
        if filename.endswith('.npy'):  # Adjust the extension as needed

            np_filename = filename.split('.')[0]

            # Create a NumPy array
            data = np.load(dir+filename)

            # Specify the HDF5 file name
            h5_file_name = outputdir+np_filename + ".h5"

            # Open the HDF5 file in write mode
            with h5py.File(h5_file_name, "w") as hdf5_file:
                # Create a dataset in the HDF5 file and write the NumPy array to it
                hdf5_file.create_dataset("note_template", data=data)


def rename_notes():
    dir = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/data_persisted/STFT/4096/tmp_W/"
    outputdir = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/single_notes/"

    contents = os.listdir(dir)

    for filename in contents:
        if filename.endswith('.npy') and filename.split('_')[0] == 'W':  # Adjust the extension as needed

            np_filename = filename.split('.')[0]

            # Create a NumPy array
            data = np.load(dir+filename)

            # Specify the HDF5 file name
            h5_file_name = outputdir+np_filename + ".h5"

            # Open the HDF5 file in write mode
            with h5py.File(h5_file_name, "w") as hdf5_file:
                # Create a dataset in the HDF5 file and write the NumPy array to it
                hdf5_file.create_dataset("dictionary", data=data)


rename_notes()

