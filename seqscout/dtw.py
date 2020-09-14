import numpy as np
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from seqscout.utils import read_rocket_league_data
import seqscout.conf as conf


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def dtw_multi(a, b):
    # we adapt DTW to the multivariate case by taking the mean of all distances between different timeseries in a and b
    sum_distance = 0
    manhattan_distance = lambda x, y: np.abs(x - y)

    for numeric, _ in a.items():
        dist, _, _, _ = dtw(a[numeric], b[numeric], manhattan_distance)
        sum_distance += dist

    return sum_distance / len(a)


def knn(X_train, X_test, y_train, k):
    # [(dist, class) ...]
    pred = []
    for x_test in X_test:
        top_k = [(float('inf'), -1)]
        best_class, min_dist = -1, float('inf')
        for i, x_train in enumerate(X_train):
            dist = dtw_multi(x_test, x_train)
            if dist < min_dist:
                min_dist = dist
                best_class = y_train[i]

        for i in range(len(top_k)):
            if top_k[i][0] > min_dist:
                top_k.insert(i, (min_dist, best_class))
                if len(top_k) > k:
                    top_k.pop()
                break

        histo = {}
        for _, class_value in top_k:
            histo[class_value] = histo.setdefault(class_value, 0) + 1

        pred.append(max(histo, key=histo.get))

    return pred


def launch_knn_dtw():
    # 1-NN
    k = 1

    # we transform data to multivariate time series of form [class, {'timeserie_name': np_array}]
    DATA = read_rocket_league_data()
    DATA_FORMATED = []

    # process time, and Goals in events

    # extracting all numerics and possible booleans
    events_all, numerics_all = set(), set()
    for line in DATA:
        for events, numerics in line[1:]:
            for event in events:
                events_all.add(event)
            for numeric, _ in numerics.items():
                numerics_all.add(numeric)

    for line in DATA:
        class_name, sequence = line[0], line[1:]
        new_line = [class_name]

        timeseries = {name: [0 for _ in range(len(sequence))] for name in events_all}
        timeseries.update(
            {name: [0 for _ in range(len(sequence))] for name in numerics_all})

        for i, (events, numerics) in enumerate(sequence):
            for event in events:
                timeseries[event][i] = 1
            for numeric, value in numerics.items():
                timeseries[numeric][i] = value

        new_line.append(timeseries)
        DATA_FORMATED.append(new_line)

    X, y = [], []

    # we transform them to numpy arrays
    for line in DATA_FORMATED:
        y.append(line[0])
        for i, (timeserie, values) in enumerate(line[1].items()):
            line[1][timeserie] = array(values).reshape(-1, 1)
        X.append(line[1])

    # normalize data
    numerics_all = {}

    # we concate all timeseries in a big one
    for line in X:
        for key, array_values in line.items():
            if key not in numerics_all:
                numerics_all[key] = array_values
            else:
                numerics_all[key] = np.concatenate((numerics_all[key], array_values), axis=None)
    # we take the mean and the standard deviation
    means_stds = {}
    for key, value in numerics_all.items():
        means_stds[key] = (np.mean(value), np.std(value))

    # we normalize
    for line in X:
        for key, array_values in line.items():
            line[key] = line[key].astype(float)
            line[key] -= means_stds[key][0]
            line[key] /= means_stds[key][1]



    # KNN step s
    FOLD_NUMBER = conf.CROSS_VALIDATION_NUMBER

    skf = StratifiedKFold(n_splits=FOLD_NUMBER)
    skf.get_n_splits(X, y)

    mean_accuracy = 0

    for train_index, test_index in skf.split(X, y):
        X_train = [x for i, x in enumerate(X) if i in train_index]
        y_train = [y for i, y in enumerate(y) if i in train_index]
        X_test = [x for i, x in enumerate(X) if i in test_index]
        y_test= [y for i, y in enumerate(y) if i in test_index]

        y_pred = knn(X_train, X_test, y_train, 5)

        accuracy_iter = accuracy_score(y_pred, y_test)
        print("Test score {}".format(accuracy_iter))
        print(confusion_matrix(y_pred, y_test))

        mean_accuracy += accuracy_iter

    mean_accuracy /= FOLD_NUMBER

    print("The mean accuracy is {}".format(mean_accuracy))
    return mean_accuracy

if __name__ == '__main__':
    launch_knn_dtw(1)
