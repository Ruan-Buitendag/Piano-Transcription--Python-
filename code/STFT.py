import soundfile as sf
import numpy as np

from scipy import signal
import librosa as lr
from numba import jit


class STFT:
    """ A class containing the stft coefficients and important values related to the STFT of a signal, channelwise """

    def __init__(self, path, time=None, channel='Sum', temporal_frame_size=64 / 1000, num_bins = 4096):
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

        if time != None:
            the_signal = the_signal[0:time * sampling_rate_local, :]

        if channel == 'Sum':
            the_signal = the_signal[:, 0] + the_signal[:, 1]
        elif channel == 'Average':
            the_signal = (the_signal[:, 0] + the_signal[:, 1]) / 2
        else:
            the_signal = the_signal[:, channel]


        frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
                                                     nperseg=num_bins,
                                                     nfft=num_bins*2, noverlap=num_bins - 882)
            # frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
            #                                              nperseg=2048,
            #                                              nfft=4096, noverlap=2048 - 882)


        # else:
        #     frequencies, time_atoms, coeff = signal.stft(the_signal, fs=sampling_rate_local,
        #                                                  nperseg=int(sampling_rate_local * temporal_frame_size),
        #                                                  nfft=int(sampling_rate_local * temporal_frame_size))
        self.time_bins = time_atoms
        self.freq_bins = frequencies
        self.sampling_rate = sampling_rate_local
        self.stft_coefficients = coeff

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

        if threshold == None:
            return np.abs(self.stft_coefficients)
        else:
            spec = np.abs(self.stft_coefficients)
            spec[spec < threshold] = 0

            # Other version, potentially helpful
            # spec = np.where(spec < np.percentile(spec, 99), 0, spec) # Forcing saprsity by keeping only the highest values

            return spec

    def get_power_spectrogram(self, threshold=None):
        """
        Computes the power spectrogram of the STFT

        Parameters
        ----------
        self: the STFT

        threshold: float
            Threshold under which values will be set to zero, for denoizing

        Returns
        -------
        spec: array
            Power Spectrogram of the STFT: array of the squared magnitudes of the STFT complex coefficients
        """

        if threshold == None:
            return np.abs(self.stft_coefficients) ** 2
        else:
            spec = np.abs(self.stft_coefficients) ** 2
            spec_zero = spec[spec < threshold] = 0
            return spec_zero

    # @jit(nopython=True)
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
        lower = self.f_pitch(p - (0.5/bins_per_note), pitch_ref, freq_ref)
        upper = self.f_pitch(p + (0.5/bins_per_note), pitch_ref, freq_ref)
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
            k = self.pool_pitch(p/bins_per_note, Fs, Y.shape[0]*2)
            Y_LF[p, :] = Y[k, :].sum(axis=0)
        F_coef_pitch = np.arange(total_bins)
        return Y_LF, F_coef_pitch

