import numpy as np

import STFT

song = "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M23_AkPnBcht.wav"

aaa = STFT.STFT(song)
s = aaa.get_mel_spec()

o = 0