import soundfile as sf
import numpy as np

from scipy import signal
import librosa as lr
from numba import jit
import matplotlib.pyplot as plt

import stft_1st_principles


class STFT:
    """ A class containing the stft coefficients and important values related to the STFT of a signal, channelwise """

    def __init__(self, path, time=None, channel='Sum', temporal_frame_size=64 / 1000, num_bins=4096, hop_length=882):
        """
        STFT of a temporal signal, given a path

        Parameters
        ----------
        path: String
            Path of the signal to evaluate
        time: None or integer
            Time value (in seconds) to crop the song:
            allows to evaluate the excerpt of the sample from 0 to time seconds.
            Set to None if the entire sample should be evaluated
            Default: None
        channel: integer
            Channel of the signal on which to perform STFT
        temporal_frame_size: float
            Size of the window to perfom STFT
            Default: 0.064 (64ms)

        Attributes
        ----------
        time_bins: array
            Time bins of the STFT
        freq_bins: array
            Frequency bins of the STFT
        sampling_rate: float
            Sampling rate of the STFT
        stft_coefficients: array
            Complex coefficiens of the STFT
        """
        # For now, this function returns the stft of only one channel
        the_signal, sampling_rate_local = sf.read(path)

        # time = 1

        if time != None:
            the_signal = the_signal[0:time * sampling_rate_local, :]

        # the_signal = the_signal[0:44100, :]

        if channel == 'Sum':
            the_signal = the_signal[:, 0] + the_signal[:, 1]
        elif channel == 'Average':
            the_signal = (the_signal[:, 0] + the_signal[:, 1]) / 2
        else:
            the_signal = the_signal[:, channel]

        # Removing the zeros at the beginning of the signal
        # counter = 0
        # while the_signal[counter] == 0:
        #     counter += 1
        #
        # the_signal = the_signal[counter+1:]

        mel_spect = None

        the_signal = the_signal / np.max(the_signal)

        # mel_spect = lr.feature.melspectrogram(y=the_signal, sr=sampling_rate_local, n_fft=num_bins * 2,
        #                                       hop_length=hop_length, n_mels=1024)
        # mel_spect = lr.power_to_db(mel_spect, ref=np.max)

        frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
                                                     nperseg=num_bins,
                                                     nfft=num_bins * 2, noverlap=num_bins - hop_length)

        self.my_stft = stft_1st_principles.stft_basic_real(the_signal, 4096, 882, nfft=4096 * 2)

        # frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
        #                                              nperseg=2048,
        #                                              nfft=4096, noverlap=2048 - 882)

        # else:
        #     frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
        #                                                  nperseg=int(sampling_rate_local * temporal_frame_size),
        #                                                  nfft=int(sampling_rate_local * temporal_frame_size))
        self.time_bins = time_atoms
        self.sampling_rate = sampling_rate_local
        self.stft_coefficients = coeff
        self.mel_spec = mel_spect

    def get_mel_spec(self):
        return self.mel_spec

    def get_magnitude_spectrogram(self, threshold=None):
        """
        Computes the magnitude spectrogram of the STFT

        Parameters
        ----------
        self: the STFT

        threshold: float
            Threshold under which values will be set to zero, for denoizing

        Returns
        -------
        spec: array
            Magnitude Spectrogram of the STFT: array of the magnitudes of the STFT complex coefficients
        """

        spec = np.abs(self.stft_coefficients)
        mag_spec = spec / np.max(spec)
        self.mag_spec = mag_spec


        # return mag_spec
        return self.my_stft

    def getDelayIndex(self):
        column = 0

        while np.all(self.mag_spec[:, column] < 0.05):
            column += 1

        return column

    def getDelay(self):
        column = 0

        while np.all(self.mag_spec[:, column] < 0.05):
            column += 1

        delay = column * (self.time_bins[1]-self.time_bins[0])
        return delay

    def f_pitch(self, p, pitch_ref=69, freq_ref=440.0):
        """Computes the center frequency/ies of a MIDI pitch

        Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

        Args:
            p (float): MIDI pitch value(s)
            pitch_ref (float): Reference pitch (default: 69)
            freq_ref (float): Frequency of reference pitch (default: 440.0)

        Returns:
            freqs (float): Frequency value(s)
        """
        return 2 ** ((p - pitch_ref) / 12) * freq_ref

    # @jit(nopython=True)
    def pool_pitch(self, p, Fs, N, pitch_ref=69, freq_ref=440.0, bins_per_note=1):
        """Computes the set of frequency indices that are assigned to a given pitch

        Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

        Args:
            p (float): MIDI pitch value
            Fs (scalar): Sampling rate
            N (int): Window size of Fourier fransform
            pitch_ref (float): Reference pitch (default: 69)
            freq_ref (float): Frequency of reference pitch (default: 440.0)

        Returns:
            k (np.ndarray): Set of frequency indices
        """
        lower = self.f_pitch(p - (0.5 / bins_per_note), pitch_ref, freq_ref)
        upper = self.f_pitch(p + (0.5 / bins_per_note), pitch_ref, freq_ref)
        k = np.arange(N // 2 + 1)
        k_freq = k * Fs / N  # F_coef(k, Fs, N)
        mask = np.logical_and(lower <= k_freq, k_freq < upper)
        return k[mask]

    # @jit(nopython=True)
    def compute_spec_log_freq(self, Y, Fs):
        """Computes a log-frequency spectrogram

        Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

        Args:
            Y (np.ndarray): Magnitude or power spectrogram
            Fs (scalar): Sampling rate
            N (int): Window size of Fourier fransform

        Returns:
            Y_LF (np.ndarray): Log-frequency spectrogram
            F_coef_pitch (np.ndarray): Pitch values
        """
        bins_per_note = 2
        total_bins = bins_per_note * 128
        Y_LF = np.zeros((total_bins, Y.shape[1]))
        for p in range(21, total_bins):
            k = self.pool_pitch(p / bins_per_note, Fs, Y.shape[0] * 2)
            Y_LF[p, :] = Y[k, :].sum(axis=0)
        F_coef_pitch = np.arange(total_bins)
        return Y_LF, F_coef_pitch
