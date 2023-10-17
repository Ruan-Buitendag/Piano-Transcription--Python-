import numpy as np

import STFT

import numpy


def getEQCurveFromRecording():
    aaa = STFT.STFT("MAPS_MUS-alb_se3_AkPnBcht.wav", 5, "Average")

    aa = aaa.getDelay()

    aaa = aaa.get_magnitude_spectrogram()[:, round(aa/0.02):]


    bbb = STFT.STFT("MAPS_MUS-alb_se3_AkPnBcht-bbbbb.wav", 5, "Average")

    bb = bbb.getDelay()

    bbb = bbb.get_magnitude_spectrogram()[:, round(bb/0.02):]

    ccc = bbb[:, :200] - aaa[:, :200]

    cccc = numpy.mean(ccc, axis=1)

    # np.save("test", cccc)

    a = -cccc+1


    return a

def getEQCurveFromSine():
    # aaa = STFT.STFT("C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/LIVE/sweep.wav", 5, "Average")
    #
    # aa = aaa.getDelay()
    #
    # aaa = aaa.get_magnitude_spectrogram()[:, round(aa/0.02):]
    # aaa /= np.max(aaa)
    #
    # bbb = STFT.STFT("C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/LIVE/Middle 3 loud aligned.wav", 5, "Average")
    #
    # bb = bbb.getDelay()
    #
    # bbb = bbb.get_magnitude_spectrogram()[:, int(bb/0.02):]
    # bbb /= np.max(bbb)
    #
    # ccc = bbb - aaa[:, :bbb.shape[1]]
    #
    # cccc = numpy.sum(ccc, axis=1)
    #
    # d = -cccc*1 + 1
    #
    # # np.save("test", cccc)
    #
    # a = 0

    aaa = STFT.STFT(
        "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/LIVE/sweep.wav", 5,
        "Average")

    aaa = aaa.get_magnitude_spectrogram()

    bbb = STFT.STFT(
        "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/LIVE/Middle 3 loud aligned.wav",
        5, "Average")

    bbb = bbb.get_magnitude_spectrogram()

    ccc = aaa-bbb

    cccc = numpy.sum(ccc, axis=1)

    d = cccc * 0.3 + 1

    d /= np.max(d)

    # np.save("test", cccc)

    a = 0


    return d

# getEQCurveFromSine()