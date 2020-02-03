import math
import random
import copy
import json

from seqscout import conf


def increase_it_number():
    global ITERATION_NUMBER
    ITERATION_NUMBER += 1


def pattern_mutable_to_immutable(pattern):
    return tuple([tuple([frozenset(i[0]), tuple(sorted([(key, tuple(value)) for key, value in i[1].items()]))]) for i in
                  pattern])


def sequence_mutable_to_immutable(sequence):
    """
    :param sequence:
    :return: the same sequence in its immutable form
    """
    return tuple([tuple([frozenset(i[0]), tuple(sorted(i[1].items()))]) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    """
    :param sequence:
    :return: the same sequence in its mutable form
    """
    return [[set(i[0]), {key: value for key, value in i[1]}] for i in sequence]


def immutable_seq(sequence):
    """
    :param sequence: a seq wich is mutable or not
    :return: an immutable seq
    """
    if type(sequence) == list:
        return sequence_mutable_to_immutable(sequence)
    else:
        return sequence


def count_target_class_data(data, target_class):
    """
    Count the number of occurences of target_class in the data
    :param data: sequential data of for [[class, {}, {} ...], [class, {}, {}], ...]
    :param target_class: the targeted class
    :return: the count
    """
    count = 0
    for row in data:
        if row[0] == target_class:
            count += 1

    return count


def k_length(sequence):
    """
    :param sequence: the considered sequence
    :return: the length of the sequence
    """
    return sum([len(i) for i in sequence])


def is_subsequence(a, b):
    """ check if sequence a is a subsequence of b
    """
    i_a, i_b = 0, 0

    while i_a < len(a) and i_b < len(b):
        # we check if buttons are present
        if a[i_a][0].issubset(b[i_b][0]):
        # now we check if numeric value is inside interval
            if all([value >= a[i_a][1][numeric][0] and value <= a[i_a][1][numeric][1] for numeric, value in
                    b[i_b][1].items()]):
                i_a += 1
        i_b += 1

    return i_a == len(a)


def subsequence_indices(a, b):
    """ Return itemset indices of b that itemset of a are included in
        Precondition: a is a subset of b
    """
    index_b_mem = 0
    indices_b = []
    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                indices_b.append(index_b)
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            return indices_b

    return indices_b


def read_json_rl(filename):
    with open(filename) as f:
        data = json.load(f)

    output_data = []
    for line in data:
        for i, state in enumerate(line['sequence']):
            del state[1]['Time']
            line['sequence'][i] = [set(state[0]), state[1]]

        new_line = [line['figure']] + line['sequence']
        output_data.append(new_line)
    return output_data


def encode_data(data, item_to_encoding):
    """
    Replaces all item in data by its encoding
    :param data:
    :param item_to_encoding:
    :return:
    """
    for line in data:
        for i, itemset in enumerate(line[1:]):
            encoded_itemset = set()
            for item in itemset:
                encoded_itemset.add(item_to_encoding[item])
            line[i + 1] = encoded_itemset

    return data


def decode_sequence(sequence, encoding_to_item):
    """
    Give the true values of sequence
    :param sequence: the sequence to decode in the form [{}, ..., {}]
    :return: the decoded sequence
    """
    return_sequence = []

    for i, itemset in enumerate(sequence):
        decoded_itemset = set()
        for item in itemset:
            decoded_itemset.add(encoding_to_item[item])
        return_sequence.append(decoded_itemset)
    return return_sequence


def extract_items(data):
    """
    :param data: date must be on the form [[class, {}, {}, ...], [class, {}, {}, ...]]
    :return: set of items extracted
    """
    items = set()
    for sequence in data:
        for itemset in sequence[1:]:
            for item in itemset:
                items.add(item)
    return sorted(list(items))


def print_results(results):
    sum_result = 0
    for result in results:
        pattern_display = ''
        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        sum_result += result[0]

        print('Quality: {}, Pattern: {}'.format(result[0], pattern_display))

    print('Average score :{}'.format(sum_result / len(results)))


def average_results(results):
    sum_result = 0
    for result in results:
        sum_result += result[0]

    return sum_result / len(results)


def extract_l_max(data):
    lmax = 0
    for line in data:
        lmax = max(lmax, k_length(line))
    return lmax


def compute_quality(data, subsequence, target_class, quality_measure=conf.QUALITY_MEASURE):
    support = 0
    data_supp = len(data)
    class_pattern_count = 0
    class_data_count = 0

    for sequence in data:
        current_class = sequence[0]
        sequence = sequence[1:]

        if is_subsequence(subsequence, sequence):
            support += 1
            if current_class == target_class:
                class_pattern_count += 1

        if current_class == target_class:
            class_data_count += 1

    if quality_measure == 'WRAcc':
        # we find the number of elements who have the right target_class
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return -0.25

        class_data_ratio = class_data_count / len(data)
        wracc = support / len(data) * (class_pattern_ratio - class_data_ratio)
        return wracc

    elif quality_measure == 'Informedness':
        tn = len(data) - support - (class_data_count - class_pattern_count)

        tpr = class_pattern_count / (class_pattern_count + (class_data_count - class_pattern_count))

        tnr = tn / (class_pattern_count + tn)
        return tnr + tpr - 1

    elif quality_measure == 'F1':
        try:
            class_pattern_ratio = class_pattern_count / support
        except ZeroDivisionError:
            return 0, 0
        precision = class_pattern_ratio
        recall = class_pattern_count / class_data_count
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except:
            f1 = 0
        return f1
    else:
        raise ValueError('The quality measure name is not valid')


def encode_data_pattern(DATA, patterns):
    """
    encode data in a transaction matrix, with features being the pattern, set to 0 or 1
    :param DATA:
    :param patterns:
    :return:
    """
    encoded_data = []

    for data in DATA:
        new_line = [data[0]] + [1 if is_subsequence(pattern[1], data[1:]) else 0 for pattern in patterns]
        encoded_data.append(new_line)

    return encoded_data
