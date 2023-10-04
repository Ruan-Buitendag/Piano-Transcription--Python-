import numpy as np
import os
import re
import time
import convolutive_MM as MM
import beta_divergence as div
from numba import jit
import STFT


# @jit(nopython=True)
def semi_supervised_transcribe_cnmf(path, beta, itmax, tol, W_dict, time_limit=None, H0=None, plot=False,
                                    model_AD=False, channel="Sum", num_bins=4096, spec_type="stft", skip_top=3500):
    """
    find H of real piano piece by semi-supervised NMF
    ----------------------------
    :param path: path of the piano piece
    :param beta: value of beta
    :param itmax: the maximal iteration number allowed in the computation of H
    :param tol: the maximal tolerance of relative error in the computation of H
    :param W_dict: note dictionary W
    :param time_limit: time limit
    :param H0: initialization of H
    :param plot: plot activation matrix H
    :return: activation matrix H
    """

    stft = STFT.STFT(path, time=time_limit, channel=channel, num_bins=num_bins)

    if spec_type == "stft":
        X = stft.get_magnitude_spectrogram()

        aa = np.load("test.npy")

        aa = aa

        X = X - aa[:, np.newaxis]

        X[X < 0] = 0.00000000001

        max_index_after_skip = W_dict.shape[1] - skip_top

        W_dict = W_dict[:, :max_index_after_skip, :]

        # for note in range(W_dict.shape[2]):
        # print(np.max(W_dict[:, :, note]))
        # W_dict[:, :, note] /= np.max(W_dict[:, :, note])

        W_dict = W_dict / np.max(W_dict)

        X = X[:max_index_after_skip, :]
    elif spec_type == "mspec":
        X = stft.get_mel_spec()

    # we remove firstly the columns whose contents are less than 1e-10
    columnlist = []
    for i in range(np.shape(X)[1]):
        if (X[:, i] < 1e-10).all():
            X[:, i] = 1e-10 * np.ones(np.shape(X)[0])

    # initialization of H using semi-supervised NMF of W(0)
    if H0 is None:
        # H0,_,_ = compute_H_nmf(X, itmax, beta, tol, W_dict[0,:,:])
        # H0,_,_ = compute_H_nmf(X, 50, beta, tol, W_dict[0,:,:])
        print('Initialization is done')

    H, n_iter, all_err = compute_H(X, itmax, beta, tol, W=W_dict, H0=H0)

    print('Computation done')

    return H, n_iter, all_err


def compute_H(X: np.array, itmax: int, beta: float, e: float, W, H0=None):
    """
    Computation of H when W is fixed
    ---------
    :param X: STFT matrix
    :param itmax: max iteration number
    :param beta: coefficient beta
    :param e: tolerance
    :param W: note template
    :param H0: initialization of activation matrix H

    :return: activation matrix H
    """
    r = np.shape(W)[2]
    ncol = np.shape(X)[1]
    T = np.shape(W)[0]

    if H0 is None:

        def gaussian(x, y, amplitude, mean_x, mean_y, sigma_x, sigma_y):
            exponent = -((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2))
            return amplitude * np.exp(exponent)

        # Define the parameters for the 2D Gaussian
        amplitude = 1.0  # Amplitude of the Gaussian
        mean_x = 0.0  # Mean (center) along the x-axis
        mean_y = 0.0  # Mean (center) along the y-axis
        sigma_x = 1.0  # Standard deviation along the x-axis
        sigma_y = 1.0  # Standard deviation along the y-axis

        # Create a grid of x and y values
        x = np.linspace(-5, 5, r)  # Adjust the range and resolution as needed
        y = np.linspace(-5, 5, ncol)  # Adjust the range and resolution as needed

        # Create a 2D matrix to store the Gaussian data
        gaussian_matrix = np.zeros((len(x), len(y)))

        # Calculate the Gaussian values and fill the matrix
        for i in range(len(x)):
            for j in range(len(y)):
                gaussian_matrix[i, j] = gaussian(x[i], y[j], amplitude, mean_x, mean_y, sigma_x, sigma_y)

        H = gaussian_matrix


        # H = np.ones([r, ncol])

        # H = np.random.rand(r, ncol)

    else:
        H = np.copy(H0)

    # set value of gamma
    if beta < -1:
        gamma = (2 - beta) ** (-1)
    elif beta > 2:
        gamma = (beta - 1) ** (-1)
    else:
        gamma = 1

    n_iter = 0

    # bbbb = np.zeros([88, 10])
    #
    # for i in range(88):
    #     for j in range(10):
    #         bbbb[i, j] = i+j

    # kkkkkk = np.sum(np.dot(W[t], MM.shift(bbbb, t)) for t in range(T))

    err_int = div.beta_divergence(beta, X, np.sum(np.dot(W[t], MM.shift(H, t)) for t in range(T)))
    print('Initial cost function: ', err_int)
    obj_previous = 0
    all_err = [err_int]

    denom_all_col = np.sum(np.dot(W[t].T, np.ones([W.shape[1], ncol])) for t in
                           range(T))

    denoms_cropped_for_end = []
    for j in range(1, T + 1):
        # tab = np.sum(np.dot(W[i].T, np.ones(W[i].shape[0])) for i in range(j))
        tab = np.sum(np.sum(W[i], axis=0) for i in range(j))

        denoms_cropped_for_end.append(tab)

    hhhhhh = np.array([denoms_cropped_for_end])

    while n_iter < itmax:
        # update H
        A = np.sum(np.dot(W[t], MM.shift(H, t)) for t in range(T))

        A[A == 0] = 1e-10

        X_hadamard_A = X * (A ** (beta - 2))
        X_hadamard_A_padded = np.concatenate((X_hadamard_A, np.zeros([W.shape[1], T])), axis=1)

        # Only a loop on T
        num = np.zeros((r, X_hadamard_A.shape[1]))
        for t in range(T):
            num = num + np.transpose(W[t]) @ X_hadamard_A_padded[:, t:ncol + t]

        H[:, :ncol - T] = H[:, :ncol - T] * (num[:, :ncol - T] / denom_all_col[:, :ncol - T]) ** gamma
        # Special case for the end, when the denominator changes
        for n in range(ncol - T, ncol):
            H[:, n] = H[:, n] * (num[:, n] / denoms_cropped_for_end[ncol - n - 1]) ** gamma

        obj = div.beta_divergence(beta, X, np.sum(np.dot(W[t], MM.shift(H, t)) for t in range(T)))

        print('cost function: ', obj)

        all_err.append(obj)
        # print('cost function: ', obj)
        # no need to update W
        # we track the relative error between two iterations
        if np.abs(obj - obj_previous) / err_int < e:
            print("Converged sufficiently")
            # break
        obj_previous = obj
        # Counter incremented here
        n_iter = n_iter + 1

    return H, n_iter, all_err

# a = (np.arange(25) + 1).reshape(5, 5).astype(np.float32)


# print(compute_H(a, 10, 1, 1e-10, np.array([a, a, a, a, a])))
