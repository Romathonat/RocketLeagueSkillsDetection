import numpy as np

def normalize(serie):
    mean = np.mean(serie)
    std = np.std(serie)

    return (serie - mean) / std
