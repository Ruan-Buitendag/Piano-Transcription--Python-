import numpy as np
import transforms
import STFT

# song = "C:\\Users\\ruanb\\OneDrive\\Desktop\\TransSSCNMF-main\\TransSSCNMF-main\\MAPS\\AkPnBcht\\ISOL\\NO\\MAPS_ISOL_NO_F_S0_M25_AkPnBcht.wav"
song = "../MAPS/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S1_M25_AkPnBcht.wav"

# transforms.CQT(path=song)
a = STFT.STFT(song)
b = a.get_magnitude_spectrogram()
c, _ = a.compute_spec_log_freq(b, 44100)
o = 0