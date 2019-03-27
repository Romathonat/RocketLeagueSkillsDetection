import numpy as np
import pandas as pd


# NB: This version of paa is exact, but is computationaly expensive
def paa_perfect(serie, w):
    '''
    :param serie: a pandas dataframe indexed on timestamp, with values as first column
    :param w:
    :return: the paa in a panda dataframe, indexed on timestamp. The timestamp corresponds to the first point of each slot
    '''
    # res = np.zeros(w)

    res = [0] * w

    n = len(serie)
    time_indexes = [None] * w

    # mean on each slot. If the last last slot contains less than w elements, we split values
    for i in range(w * n):
        idx = i // n
        pos = i // w

        # np.add.at(res, idx, serie[pos])
        res[idx] += serie[pos]

        if time_indexes[idx] is None:
            time_indexes[idx] = serie.index.values[pos]

    values = [x / n for x in res]
    return pd.Series(values, index=time_indexes)

    # we need to assign a timestamp to each slot: we take the first one


def paa(serie, w):
    '''
    If w is not a multiple of len(serie), the last element is just shorter (if len(n) >> w) and we have a sequence of
    len w+1 (computationaly easier)
    :param serie: a pandas dataframe indexed on timestamp, with values as first column
    :param w:
    :return: the paa in a panda dataframe, indexed on timestamp. The timestamp corresponds to the first point of each slot
    '''
    n = len(serie)
    res = [0] * (w + 1)
    slot_length = n // w

    time_indexes = [None] * (w + 1)

    for i in range(n):
        idx = i // slot_length

        if idx > w:
            # we are in this case if we arrive at the last element
            idx = w

        res[idx] += serie[i]

        if time_indexes[idx] is None:
            time_indexes[idx] = serie.index.values[i]

    values = [x / slot_length for x in res[:-1]]

    # case where w is a mutliple of len(serie)
    if n % w == 0:
        res = res[:-1]
        time_indexes = time_indexes[:-1]
    else:
        # the last value needs to be divided by n % w
        values.append(res[-1] / (n % w))

    return pd.Series(values, index=time_indexes)

# print(paa([1, 2, 0, 4, 3, 5, 6, -2, 3  -4], 4))
# print(paa([1, 2, 3], 1))
