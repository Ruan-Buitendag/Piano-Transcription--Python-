import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa as lr
import soundfile as sf
import STFT


def stft_basic_real(x, window_length, H=8, only_positive_frequencies=False, fs=44100, nfft=4096, time_limit=10):
    sig = np.copy(x)

    sig = sig[:time_limit * 44100]

    sig = np.append(np.zeros(window_length // 2), sig)
    sig = np.append(sig, np.zeros(window_length // 2))

    npad = 0

    while (sig.shape[0] + npad - window_length) % (H) != 0:
        npad += 1

    # npad = 0

    sig = np.append(sig, np.zeros(npad))

    w = np.hanning(window_length)

    N = window_length
    L = len(sig)

    # how many hops can we do
    M = int(np.floor((L - N) / H)) + 1

    X_mag = np.zeros((nfft // 2 + 1, M))

    for m in range(M):
        x_win = sig[m * H:m * H + N] * w
        x_win = np.append(x_win, np.zeros(nfft - x_win.shape[0]))
        X_win = np.fft.fft(x_win)

        X_mag[:, m] = np.abs(X_win[0:nfft // 2 + 1])

    X_mag /= np.sum(w)

    X_mag /= np.max(X_mag)

    return X_mag


the_signal, sampling_rate_local = sf.read(
    "C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M23_AkPnBcht.wav")

# a = the_signal[:, 0]
the_signal = (the_signal[:, 0] + the_signal[:, 1]) / 2

d = stft_basic_real(the_signal, 4096, H=882, fs=sampling_rate_local, nfft=4096*2, time_limit=1)

# aaa = STFT.STFT("C:/Users/ruanb/OneDrive/Desktop/Piano Transcripton/Piano transcription/MAPS/AkPnBcht/ISOL/NO/MAPS_ISOL_NO_F_S0_M23_AkPnBcht.wav", time=1, num_bins=4096)
# bbb = aaa.get_magnitude_spectrogram()


a = 0
