import os
import struct

import h5py
import csv

import numpy as np


def write_notes_to_csv():
    path = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/single_notes/hdf5/"

    outputdir = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/single_notes/csv/"

    contents = os.listdir(path)

    for filename in contents:
        if filename.endswith('.h5'):  # Adjust the extension as needed

            filename_wo_extension = filename.split('.')[0]

            with h5py.File(path + filename, 'r') as h5file:
                # Read data from the HDF5 file here
                dataset = h5file['dictionary']
                data = dataset[:]

                T, bins, notes = data.shape

                file_path = outputdir + filename_wo_extension + '.csv'
                with open(file_path, 'w', newline='') as csvfile:
                    # Create a CSV writer object
                    csv_writer = csv.writer(csvfile)

                    csv_writer.writerow([T])
                    csv_writer.writerow([bins])
                    csv_writer.writerow([notes])

                    for t in range(T):
                        for b in range(bins):
                            for n in range(notes):
                                csv_writer.writerow([data[t][b][n]])

                    csvfile.close()


def write_dict_to_csv(piano):
    file_path = 'C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/dictionaries/conv_dict_piano_' + piano + '.h5'
    with h5py.File(file_path, 'r') as h5file:
        # Read data from the HDF5 file here
        dataset = h5file['dictionary']
        data = dataset[:]

        T, bins, notes = data.shape

        file_path = 'C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano Transcription (C)/data_persisted/dictionaries/conv_dict_piano_' + piano + '.csv'
        with open(file_path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow([T])
            csv_writer.writerow([bins])
            csv_writer.writerow([notes])

            for t in range(T):
                for b in range(bins):
                    for n in range(notes):
                        csv_writer.writerow([data[t][b][n]])

            csvfile.close()


def read_dict_from_csv():
    dirrrr = "C:\\Users\\ruanb\\OneDrive\\Desktop\\Piano Transcripton\\Piano Transcription (C)\\data_persisted\\single_notes\\"

    file_path = '/csv/W_one_note_piano_AkPnBcht_midi_21.csv'
    aaaaa = '/bin/W_one_note_piano_AkPnBcht_midi_21.bin'

    with open(dirrrr+file_path, 'r', newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        T = int(next(csv_reader)[0])
        bins = int(next(csv_reader)[0])
        notes = int(next(csv_reader)[0])

        data = np.zeros((T, bins, notes))

        for t in range(T):
            for b in range(bins):
                for n in range(notes):
                    data[t][b][n] = float(next(csv_reader)[0])

        csvfile.close()

        with open(dirrrr + aaaaa, "wb") as file:

            packed_data = struct.pack('i', T)
            file.write(packed_data)

            packed_data = struct.pack('i', bins)
            file.write(packed_data)

            packed_data = struct.pack('i', notes)
            file.write(packed_data)

            for t in range(T):
                for b in range(bins):
                    for n in range(notes):
                        packed_data = struct.pack('d', data[t][b][n])
                        file.write(packed_data)



            file.close()


piano_W = ["AkPnBcht", "AkPnBsdf", "AkPnCGdD", "AkPnStgb", "ENSTDkCl"]

# for piano in piano_W:
#     write_dict_to_csv(piano)
# read_dict_from_csv()

read_dict_from_csv()

# write_notes_to_csv()
