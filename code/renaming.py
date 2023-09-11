import os

# Specify the directory where your files are located
directory = '../data_persisted/STFT/4096/tmp_W'

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.npy'):  # Adjust the extension as needed
        # Split the filename into parts
        parts = filename.split('_')

        # Find the index of the "stftAD_True" part
        try:
            stft_index = parts.index('stftAD')
        except ValueError:
            print(f'File {filename} does not contain "stftAD_True"')
            continue

        # Replace the "stftAD_True" part with "stft_4096"
        parts[stft_index] = 'stft'
        parts[stft_index+1] = '4096'

        # Join the parts back together to form the new filename
        new_filename = '_'.join(parts)

        # Rename the file
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')