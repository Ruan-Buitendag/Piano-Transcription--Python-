import numpy as np

import STFT

import numpy

aaa = STFT.STFT("MAPS_MUS-alb_se3_AkPnBcht.wav", 5, "Average")

aa = aaa.getDelay()

aaa = aaa.get_magnitude_spectrogram()[:, round(aa/0.02):]




bbb = STFT.STFT("MAPS_MUS-alb_se3_AkPnBcht-bbbbb.wav", 5, "Average")

bb = bbb.getDelay()

bbb = bbb.get_magnitude_spectrogram()[:, round(bb/0.02):]

ccc = bbb[:, :200] - aaa[:, :200]

cccc = numpy.mean(ccc, axis=1)

np.save("test", cccc)

a = 0