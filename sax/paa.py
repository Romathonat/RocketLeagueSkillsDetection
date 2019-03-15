import numpy as np

def paa(serie, w):
    res = np.zeros(w)
    n = len(serie)

    if n % w == 0:
        slot_sum = n // w
        for i in range(n):
            idx = i // slot_sum
            np.add.at(res, idx, serie[i])
        return res / slot_sum
    else:
        for i in range(0, w * n):
            idx = i // n
            pos = i // w
            np.add.at(res, idx, serie[pos])

    return res / n
