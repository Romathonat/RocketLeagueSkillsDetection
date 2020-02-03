import datetime
import random
import copy
import pathlib

import math
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from seqscout.utils import sequence_mutable_to_immutable, \
    k_length, \
    count_target_class_data, extract_items, compute_quality, \
    sequence_immutable_to_mutable, encode_data, \
    read_json_rl, print_results, pattern_mutable_to_immutable, encode_data_pattern

from seqscout.priorityset import PrioritySet, PrioritySetUCB
import seqscout.conf as conf


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


def generalize_sequence(sequence, data, target_class, numerics_values=None, quality_measure=conf.QUALITY_MEASURE):
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

    # now we compute the quality measure
    quality = compute_quality(data, sequence, target_class, quality_measure=quality_measure)

    return sequence, quality


def UCB(score, Ni, N):
    # we choose C = 0.5
    return (score + 0.25) * 2 + 0.5 * math.sqrt(2 * math.log(N) / Ni)


def play_arm(sequence, data, target_class, numerics_values=None, quality_measure=conf.QUALITY_MEASURE):
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
                                           quality_measure=quality_measure)

    return pattern, quality


def seq_scout(data, target_class, numerics_values=None, time_budget=conf.TIME_BUDGET, top_k=conf.TOP_K,
              iterations_limit=conf.ITERATIONS_NUMBER, theta=conf.THETA, quality_measure=conf.QUALITY_MEASURE):
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
                                    quality_measure=quality_measure)
        pattern = pattern_mutable_to_immutable(pattern)
        sorted_patterns.add(pattern, quality)

        # we update scores
        updated_quality = (Ni * mean_quality + quality) / (Ni + 1)
        UCB_score = UCB(updated_quality, Ni + 1, N)
        UCB_scores.add(sequence, (UCB_score, Ni + 1, updated_quality))

        N += 1

    print("seqscout optimized iterations: {}".format(N))

    #return sorted_patterns.get_top_k_non_redundant(data, top_k)
    return sorted_patterns.get_top_k(top_k)


def launch():
    DATA = read_json_rl('../data/final_sequences.json')

    # shuffle data
    random.shuffle(DATA)

    # take 80% for train
    indice_split = int(0.8 * len(DATA))
    DATA_TRAIN = DATA[:indice_split]
    DATA_TEST = DATA[indice_split:]

    numerics_values = preprocess(DATA)

    patterns_immu, patterns_mutable = [], []

    # we do not use what is discriminative of -1 (other thing)
    #for i in ["1", "2", "3", "5", "6", "7"]:
    for i in ["1"]:
        patterns_immu_temp, patterns_mutable_temp = seq_scout(DATA_TRAIN, i, numerics_values=numerics_values, time_budget=10, top_k=30, iterations_limit=10000)
        patterns_immu += patterns_immu_temp
        patterns_mutable += patterns_mutable_temp

    # look for pattern for each possible class
    print_results(patterns_immu)

    '''
    encoded_data_train = np.array(encode_data_pattern(DATA_TRAIN, patterns_mutable))
    encoded_data_test = np.array(encode_data_pattern(DATA_TEST, patterns_mutable))

    Y_train, X_train = encoded_data_train[:, 0], encoded_data_train[:, 1:]
    Y_test, X_test = encoded_data_test[:, 0], encoded_data_test[:, 1:]

    Y_pred = OneVsRestClassifier(DecisionTreeClassifier(random_state=0)).fit(X_train, Y_train).predict(X_test)

    print(accuracy_score(Y_test, Y_pred))
    '''
if __name__ == '__main__':
    launch()
