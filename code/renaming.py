import os

# Specify the directory where your files are located
directory = '../data_persisted/STFT/4096/tmp_W/'

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npy'):  # Adjust the extension as needed
        # Split the filename into parts
        just_name = filename.split('.')[0]

        parts = just_name.split('_')

        new_parts = [parts[0], parts[1], parts[2], parts[3], parts[4], parts[15], parts[16]]

        # Join the parts back together to form the new filename
        new_filename = '_'.join(new_parts) + ".npy"

        # Rename the file
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')