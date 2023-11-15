import h5py
import numpy as np

import STFT

import numpy

import struct

# Sample array of floats
float_array = [1.23, 4.56, 7.89, 10.11]

# File path where you want to save the binary data
file_path = "C:\\Users\\ruanb\\OneDrive\\Documents\\build-untitled7-Old_Qt_Kit-Debug\\float_array.bin"

# Open the file in binary write mode
with open(file_path, "wb") as file:
    # Iterate through the float array and pack each float as binary data
    for value in float_array:
        packed_data = struct.pack('d', value)
        file.write(packed_data)

print(f"Array of floats written to {file_path}")
