import librosa
import numpy as np
import matplotlib.pyplot as plt


def f0_estimation(path):
    y, sr = librosa.load(path)

    # Calculate the autocorrelation function
    autocorr = np.correlate(y, y, mode='full')

    # Keep only the positive lags
    autocorr = autocorr[len(autocorr)//2:]

    # Find the peak in the autocorrelation (excluding the first lag)
    start = sr // 500  # exclude the first few samples to remove DC component
    end = sr // 50     # limit the search range for peaks
    fundamental_freq_period = np.argmax(autocorr[start:end]) + start

    # Calculate fundamental frequency in Hz
    fundamental_freq = sr / fundamental_freq_period

    return librosa.hz_to_midi(fundamental_freq)
