import datetime
import random
import copy
import pathlib

import math
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

from seqscout.utils import sequence_mutable_to_immutable, \
    k_length, \
    count_target_class_data, extract_items, compute_quality, \
    sequence_immutable_to_mutable, encode_data, \
    read_json_rl, print_results, pattern_mutable_to_immutable, encode_data_pattern

from seqscout.priorityset import PrioritySet, PrioritySetUCB
import seqscout.conf as conf
from tpot import TPOTClassifier


def filter_target_class(data, target_class):
    filter_data = []
    for line in data:
        if line[0] == target_class:
            filter_data.append(line)

    return filter_data


def get_itemset_memory(data):
    memory = set()
    for line in data:
        for itemset in line[1:]:
            memory.add(frozenset(itemset))
    return memory


def preprocess(DATA):
    """
    :param DATA
    :return: a dict, containing for each variable the ordered list of all its possible values
    """
    numerics_values = {}
    for line in DATA:
        for state in line[1:]:
            buttons, numerics = state
            for numeric, value in numerics.items():
                numerics_values.setdefault(numeric, set()).add(value)

    # now we transform those set to sorted list (not very optimised, should do it in the for loop)
    for numeric, numeric_set in numerics_values.items():
        numerics_values[numeric] = sorted(numeric_set)

    return numerics_values


def is_included(pattern, pattern_set):
    if pattern in pattern_set:
        return True
    else:
        for x in pattern_set:
            if pattern.issubset(x):
                return True
        return False


def find_i_value(numeric_values, value):
    """
    Find the position of value in numeric_values
    :param numeric_values:
    :param value:
    :return:
    """
    for i, value_numeric in enumerate(numeric_values):
        if value_numeric == value:
            return i
    return -1


def generalize_sequence(sequence, data, target_class, numerics_values=None, quality_measure=conf.QUALITY_MEASURE, numeric_remove_proba=conf.NUMERIC_REMOVE_PROBA):
    sequence = copy.deepcopy(sequence)

    # we remove z items randomly among buttons
    seq_items_nb = len([i for j_set in sequence for i in j_set[0]])

    z = random.randint(0, seq_items_nb - 1)

    for _ in range(z):
        chosen_itemset_i = random.randint(0, len(sequence) - 1)
        chosen_itemset = sequence[chosen_itemset_i][0]

        chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

        if len(chosen_itemset) == 0:
            sequence.pop(chosen_itemset_i)

    # for numerics, we take a random element to the right, and a random to the left (considering the list is sorted)
    for _, numerics in sequence:
        for numeric, value in numerics.items():
            i_value = find_i_value(numerics_values[numeric], value)

            if i_value == 0:
                left_value = 0
            else:
                left_value = random.sample(numerics_values[numeric][:i_value], 1)[0]

            if i_value == len(numerics_values[numeric]) - 1:
                rigth_value = len(numerics_values[numeric]) - 1
            else:
                rigth_value = random.sample(numerics_values[numeric][i_value + 1:], 1)[0]
            numerics[numeric] = [left_value, rigth_value]

            # with numeric_remove_proba chance we remove this numeric !
            if random.random() < numeric_remove_proba:
                numerics[numeric] = [-float('inf'), float('inf')]

    # now we compute the quality measure
    quality = compute_quality(data, sequence, target_class, quality_measure=quality_measure)

    return sequence, quality


def UCB(score, Ni, N):
    # we choose C = 0.5
    return (score + 0.25) * 2 + 0.5 * math.sqrt(2 * math.log(N) / Ni)


def play_arm(sequence, data, target_class, numerics_values=None, quality_measure=conf.QUALITY_MEASURE, numeric_remove_proba=conf.NUMERIC_REMOVE_PROBA):
    '''
    Select object, generalise
    :param sequence: immutable sequence to generalise
    :param data:
    :param data_target_class: elements of the data with target class
    :return:
    '''
    sequence = sequence_immutable_to_mutable(sequence)

    pattern, quality = generalize_sequence(sequence,
                                           data,
                                           target_class,
                                           numerics_values=numerics_values,
                                           quality_measure=quality_measure,
                                           numeric_remove_proba=numeric_remove_proba)

    return pattern, quality


def seq_scout(data, target_class, numerics_values=None, time_budget=conf.TIME_BUDGET, top_k=conf.TOP_K,
              iterations_limit=conf.ITERATIONS_NUMBER, theta=conf.THETA, quality_measure=conf.QUALITY_MEASURE, numeric_remove_proba=conf.NUMERIC_REMOVE_PROBA):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    data_target_class = filter_target_class(data, target_class)
    sorted_patterns = PrioritySet(k=top_k, theta=theta)
    UCB_scores = PrioritySetUCB()

    N = 1

    # init: we add objects with the best ucb so that they are all played one time in the main procedure.
    # By putting a null N, we ensure the mean of the quality will be correct
    for sequence in data_target_class:
        sequence_i = sequence_mutable_to_immutable(sequence[1:])
        UCB_score = UCB(float("inf"), 1, N)
        UCB_scores.add(sequence_i, (UCB_score, 0, 0))

    # play with time budget
    while datetime.datetime.utcnow() - begin < time_budget and N < iterations_limit:
        # we take the best UCB
        _, Ni, mean_quality, sequence = UCB_scores.pop()

        pattern, quality = play_arm(sequence, data, target_class, numerics_values=numerics_values,
                                    quality_measure=quality_measure, numeric_remove_proba=numeric_remove_proba)
        pattern = pattern_mutable_to_immutable(pattern)
        sorted_patterns.add(pattern, quality)

        # we update scores
        updated_quality = (Ni * mean_quality + quality) / (Ni + 1)
        UCB_score = UCB(updated_quality, Ni + 1, N)
        UCB_scores.add(sequence, (UCB_score, Ni + 1, updated_quality))

        N += 1

    # print("seqscout iterations: {}".format(N))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)
    # return sorted_patterns.get_top_k(top_k)


def launch():
    # EXTRACTING PATTERNS
    DATA = read_json_rl('../data/final_sequences.json')

    # REMOVING BUTTONS FOR TESTING
    # DATA = [[data[0], [{1}, data[1][1]]] for data in DATA]

    # shuffle data
    random.shuffle(DATA)

    # take 80% for train
    indice_split = int(0.8 * len(DATA))
    DATA_TRAIN = DATA[:indice_split]
    DATA_TEST = DATA[indice_split:]

    numerics_values = preprocess(DATA)

    patterns_mutable = []

    # we do not use what is discriminative of -1 (other thing)
    for i in ["1", "2", "3", "5", "6", "7"]:
    #for i in ["1"]:
        patterns_mutable_temp = seq_scout(DATA_TRAIN, i, numerics_values=numerics_values, time_budget=1000,
                                          iterations_limit=5000, top_k=20)
        patterns_mutable += patterns_mutable_temp

        # look for pattern for each possible class
    print_results(patterns_mutable)

    encoded_data_train = np.array(encode_data_pattern(DATA_TRAIN, patterns_mutable))
    encoded_data_test = np.array(encode_data_pattern(DATA_TEST, patterns_mutable))

    Y_train, X_train = encoded_data_train[:, 0], encoded_data_train[:, 1:]
    Y_test, X_test = encoded_data_test[:, 0], encoded_data_test[:, 1:]

    np.save("Y_train", Y_train)
    np.save("X_train", X_train)
    np.save("Y_test", Y_test)
    np.save("X_test", X_test)

    X_train = np.load("X_train.npy")
    Y_train = np.load("Y_train.npy")
    X_test = np.load("X_test.npy")
    Y_test = np.load("Y_test.npy")

    X_train = X_train.astype(int)
    Y_train = Y_train.astype(int)
    X_test = X_test.astype(int)
    Y_test = Y_test.astype(int)

    clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=0)).fit(X_train, Y_train)
    #clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_estimators=100, max_depth=6)).fit(X_train, Y_train)
    # clf = OneVsRestClassifier(xgb.XGBClassifier()).fit(X_train, Y_train)

    Y_pred_proba = clf.predict_proba(X_test)
    Y_pred = clf.predict(X_test)

    # some stats
    classes = np.unique(Y_train)
    print(classes)

    sum_diff_one_by_n = {key: 0 for key in classes}
    one_by_n = 1 / len(classes)

    for i, probas in enumerate(Y_pred_proba):
        for j in probas:
            sum_diff_one_by_n[Y_test[i]] += abs(one_by_n - j)

    print(sum_diff_one_by_n)

    print("Train score {}".format(accuracy_score(clf.predict(X_train), Y_train)))
    print("Test score {}".format(accuracy_score(Y_test, Y_pred)))
    print(confusion_matrix(Y_test, Y_pred))

def stratified_k_fold(k=conf.CROSS_VALIDATION_NUMBER, pattern_number=conf.TOP_K, iteration_limit=conf.ITERATIONS_NUMBER, classif=conf.CLASSIFICATION_ALGORITHM, numeric_remove_proba=conf.NUMERIC_REMOVE_PROBA):
    # EXTRACTING PATTERNS
    DATA = read_json_rl('../data/final_sequences.json')

    X = np.array([i[1:] for i in DATA])
    Y = np.array([[i[0]] for i in DATA])

    numerics_values = preprocess(DATA)

    skf = StratifiedKFold(n_splits=k)

    mean_accuracy = 0
    accuracy_list = []

    for i_cv, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print("Progression {}%".format((i_cv) * 100 / k))
        patterns_mutable = []

        DATA_TRAIN = [[Y[i]] + list(X[i]) for i in train_index]
        DATA_TEST = [[Y[i]] + list(X[i]) for i in test_index]

        # we do not use what is discriminative of -1 (noise)
        for i in ["1", "2", "3", "5", "6", "7"]:
            # for i in ["1"]:
            patterns_mutable_temp = seq_scout(DATA_TRAIN, i, numerics_values=numerics_values, time_budget=1000,
                                              iterations_limit=iteration_limit, top_k=pattern_number, numeric_remove_proba=numeric_remove_proba)
            patterns_mutable += patterns_mutable_temp

            # look for pattern for each possible class
        #print_results(patterns_mutable)

        encoded_data_train = np.array(encode_data_pattern(DATA_TRAIN, patterns_mutable))
        encoded_data_test = np.array(encode_data_pattern(DATA_TEST, patterns_mutable))

        Y_train, X_train = encoded_data_train[:, 0], encoded_data_train[:, 1:]
        Y_test, X_test = encoded_data_test[:, 0], encoded_data_test[:, 1:]

        Y_train = Y_train.astype(int)
        Y_test = Y_test.astype(int)

        if classif == "DT":
            clf = OneVsRestClassifier(DecisionTreeClassifier(random_state=0)).fit(X_train, Y_train)
        elif classif == "RF":
            clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_estimators=100, max_depth=6)).fit(X_train,
                                                                                                             Y_train)
        else:
            raise ValueError('Wrong classification algorithm name')

        Y_pred_proba = clf.predict_proba(X_test)
        Y_pred = clf.predict(X_test)

        # some stats
        classes = np.unique(Y_train)
        print(classes)

        sum_diff_one_by_n = {key: 0 for key in classes}
        one_by_n = 1 / len(classes)

        for i, probas in enumerate(Y_pred_proba):
            for j in probas:
                sum_diff_one_by_n[Y_test[i]] += abs(one_by_n - j)

        print(sum_diff_one_by_n)

        accuracy_iter = accuracy_score(Y_test, Y_pred)
        print("Train score {}".format(accuracy_score(clf.predict(X_train), Y_train)))
        print("Test score {}".format(accuracy_iter))
        print(confusion_matrix(Y_test, Y_pred))
        mean_accuracy += accuracy_iter
        accuracy_list.append(accuracy_iter)

    mean_accuracy /= k
    print('The mean accuracy is {}'.format(mean_accuracy))
    return mean_accuracy, accuracy_list

def cross_validate(k):
    # EXTRACTING PATTERNS
    DATA = read_json_rl('../data/final_sequences.json')

    numerics_values = preprocess(DATA)

    Y_test_all = np.array([int(data[0]) for data in DATA])
    Y_pred_all = np.zeros(Y_test_all.shape)

    for i_k in range(k):
        print("Progression {}%".format((i_k) * 100 / k))
        test_size = (int(len(DATA) / k))

        if i_k != k - 1:
            DATA_TRAIN, DATA_TEST = DATA[:i_k * test_size] + DATA[(i_k + 1) * test_size:], \
                                    DATA[i_k * test_size:(i_k + 1) * test_size]
        else:
            DATA_TRAIN, DATA_TEST = DATA[:i_k * test_size], DATA[i_k * test_size:]

        patterns = []

        # look for pattern for each possible class
        for j in ["1", "2", "3", "5", "6", "7"]:
            patterns_temp = seq_scout(DATA_TRAIN, j, numerics_values=numerics_values, time_budget=1000,
                                      iterations_limit=5000, top_k=30)
            patterns += patterns_temp

        encoded_data_train = np.array(encode_data_pattern(DATA_TRAIN, patterns))
        encoded_data_test = np.array(encode_data_pattern(DATA_TEST, patterns))

        Y_train, X_train = encoded_data_train[:, 0], encoded_data_train[:, 1:]
        Y_test, X_test = encoded_data_test[:, 0], encoded_data_test[:, 1:]

        X_train = X_train.astype(int)
        Y_train = Y_train.astype(int)
        X_test = X_test.astype(int)
        Y_test = Y_test.astype(int)

        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_estimators=100, max_depth=6)).fit(X_train, Y_train)

        for i_pred, prediction in enumerate(clf.predict(X_test)):
            Y_pred_all[i_k * test_size + i_pred] = prediction

    print("Test score {}".format(accuracy_score(Y_test_all, Y_pred_all)))
    print(confusion_matrix(Y_test_all, Y_pred_all))


if __name__ == '__main__':
    #launch()
    #cross_validate(5)
    stratified_k_fold()


# BEST CONF######
# : no redundant (not mandatory), time=inf, 1000 iterations, top-k=30, random forest -> 86%
# 1/2 no restriction on numeric, adding time since the beggining, leave-one-out

# TODO
# try with only numerics (with only buttons ?) -> good on train (1), bad on test
# Try to use the level from misere ??? -> not easy to adapt because numerics
# noise (-1) is a problem: remove it with a threshold ?
