from sktime.transformers.compose import ColumnConcatenator
from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.dictionary_based.boss import BOSSEnsemble
from sktime.classifiers.compose import ColumnEnsembleClassifier
from sktime.classifiers.shapelet_based import ShapeletTransformClassifier
from sktime.datasets import load_basic_motions, load_gunpoint
from sktime.pipeline import Pipeline

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from seqscout.utils import read_json_rl

from sktime.classifiers.compose import TimeSeriesForestClassifier


def launch_knn_boss(k):
    # we transform data in multivariate time series of form [class, {'timeserie_name': np_array}]
    DATA = read_json_rl('../data/final_sequences.json')
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
        x_line = []
        for i, (timeserie, values) in enumerate(line[1].items()):
            x_line.append(np.array(values))
        X.append(np.array(x_line))

    # KNN step
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    X_train, y_train = load_basic_motions(split='TRAIN', return_X_y=True)
    X_test, y_test = load_basic_motions(split='TEST', return_X_y=True)

    print(X_train.shape)

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    tsf = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                     n_estimators=100,
                                     criterion='entropy',
                                     bootstrap=True,
                                     oob_score=True,
                                     random_state=1)

    #y_pred = knn(X_train, X_test, y_train, 5)

    print("Test score {}".format(accuracy_score(y_pred, y_test)))
    print(confusion_matrix(y_pred, y_test))


if __name__ == '__main__':
    launch_knn_boss(1)

