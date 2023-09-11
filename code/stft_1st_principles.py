import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa as lr


def stft_basic_real(x, window_length, H=8, only_positive_frequencies=False, fs=44100, nfft=4096):
    sig = np.copy(x)

    sig = np.append(np.zeros(window_length // 2), sig)
    sig = np.append(sig, np.zeros(window_length // 2))

    npad = 0

    while (sig.shape[0] + npad - window_length) % (H) != 0:
        npad += 1

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
    # X_mag /= nfft

    # if only_positive_frequencies:
    #     K = 1 + N // 2
    #     X_mag = X_mag[0:K, :]

    return X_mag

def stft_basic(x, window_length, H=8, only_positive_frequencies=False, fs=44100, nfft=4096):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)
    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)
            :param H:
            :param window_length:
    """
    sig = np.copy(x)

    sig = np.append(np.zeros(window_length // 2), sig)
    sig = np.append(sig, np.zeros(window_length // 2))

    npad = 0

    while (x.shape[0] + npad - window_length) % (hop_size) != 0:
        npad += 1

    sig = np.append(sig, np.zeros(npad))

    w = np.hanning(window_length)

    N = window_length
    L = len(sig)

    # how many hops can we do
    M = np.floor((L - N) / H).astype(int) + 1

    X = np.zeros((nfft, M), dtype='complex')
    for m in range(M):
        x_win = sig[m * H:m * H + N] * w
        x_win = np.append(x_win, np.zeros(nfft - x_win.shape[0]))
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    X /= np.sum(w)

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]

    return X


sr = 5000
num_points = 10000
t = np.linspace(0, num_points/sr, num_points)
frequency = 440  # Frequency of the sinusoid (5 Hz)
amplitude = 1.0
sig = amplitude * np.sin(2 * np.pi * frequency * t)

n = 4096
hop_size = 882

# Compute the Short-Time Fourier Transform (STFT)
frequencies, times, Zxx = signal.stft(sig, fs=sr, nperseg=n, noverlap=n - hop_size, nfft=n*2)

aaaaa = stft_basic(sig, n, H=hop_size, only_positive_frequencies=True, nfft=n*2)
bbbbb = stft_basic_real(sig, n, H=hop_size, only_positive_frequencies=True, nfft=n*2)


fig, ax = plt.subplots(nrows=3, ncols=1)

A = np.abs(Zxx)
C = np.abs(bbbbb)

lr.display.specshow(A, sr=sr, x_axis='time', y_axis='hz', ax=ax[0])
lr.display.specshow(C, sr=sr, x_axis='time', y_axis='hz', ax=ax[2])

# ax[0].set_xlim(0, 10)
# ax[1].set_xlim(0, 10)


plt.show()

# # Plot the magnitude of the STFT
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(times, frequencies, 20 * np.log10(np.abs(Zxx)), shading='auto')
# plt.colorbar(label='Magnitude (dB)')
# plt.title('STFT of the Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.ylim(0, 10)  # Adjust the frequency range for better visualization
# plt.show()
