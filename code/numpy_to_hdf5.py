import h5py
import numpy as np

file = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/data_persisted/STFT/4096/conv_dict_piano_AkPnStgb_beta_1_T_10_init_L1_stft_4096_itmax_500_intensity_M.npy"
np_filename = file.split('/')[-1][:-4]

# Create a NumPy array
data = np.load(file)

# Specify the HDF5 file name
h5_file_name = np_filename + ".h5"

# Open the HDF5 file in write mode
with h5py.File(h5_file_name, "w") as hdf5_file:
    # Create a dataset in the HDF5 file and write the NumPy array to it
    hdf5_file.create_dataset("dictionary", data=data)



