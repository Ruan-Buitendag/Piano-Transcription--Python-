import matplotlib.pyplot
import numpy as np
import STFT

import matplotlib.pyplot as plt

# spectrogram  = STFT.STFT("../MAPS/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.wav")
spectrogram  = STFT.STFT("../MAPS/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S1_M50_AkPnBcht.wav")

mag_sepc = spectrogram.get_magnitude_spectrogram()[:1500, 25:175]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
mag_sepc[mag_sepc == 0] = 1e-10

# Plot the spectrogram
cax = ax.pcolormesh(10 * np.log10(mag_sepc), shading='auto', cmap='turbo')
plt.colorbar(cax, label='dB')

# Set axis labels and title
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Spectrogram')

# Display the plot
plt.show()
