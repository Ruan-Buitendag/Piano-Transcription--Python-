import numpy as np

from scipy import signal
import librosa as lr
from matplotlib import pyplot as plt

# make a 5x5 np array

a = (np.arange(25)+1).reshape(5, 5).astype(np.float32)

print(a ** -1)
