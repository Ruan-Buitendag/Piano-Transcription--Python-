import numpy as np

import convolutive_MM as mm

from scipy import signal
import librosa as lr
from matplotlib import pyplot as plt
import activation_functions as af
import STFT


import csv


import numpy as np

# Create a sample NumPy array
data = np.array([range(125)]).reshape((5, 5, 5))

# Specify the file path where you want to save the CSV file
file_path = "data.csv"

# Use np.savetxt() to save the NumPy array to a CSV file
np.savetxt(file_path, data, delimiter=',', fmt="%.5f")

a = 0
