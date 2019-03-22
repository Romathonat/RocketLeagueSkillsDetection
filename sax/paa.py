import numpy as np


def paa(serie, w):
    res = np.zeros(w)
    n = len(serie)

    # mean on each slot. If the last last slot contains less than w elements, we split values

    for i in range(w * n):
        idx = i // n
        pos = i // w
        np.add.at(res, idx, serie[pos])
    return res / n

# print(paa([1, 2, 0, 4, 3, 5, 6, -2, 3  -4], 4))
# print(paa([1, 2, 3], 1))
