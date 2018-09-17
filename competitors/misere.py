import datetime
import random
import copy
import math

from seqehc.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data
from seqehc.priorityset import PrioritySet


def compute_WRAcc(data, subsequence, target_class):
    subsequence_supp = 0
    data_supp = len(data)
    class_subsequence_supp = 0
    class_data_supp = 0

    for sequence in data:
        current_class = sequence[0]
        sequence = sequence[1:]

        if is_subsequence(subsequence, sequence):
            subsequence_supp += 1
            if current_class == target_class:
                class_subsequence_supp += 1

        if current_class == target_class:
            class_data_supp += 1

    return (subsequence_supp / data_supp) * (
        class_subsequence_supp / subsequence_supp -
        class_data_supp / data_supp)


def compute_WRAcc_vertical(data, subsequence, target_class, bitset_slot_size,
                           itemsets_bitsets, class_data_count, first_zero_mask,
                           last_ones_mask):

    length = k_length(subsequence)
    bitset = 0

    if length == 0:
        # the empty node is present everywhere
        # we just have to create a vector of ones
        bitset = 2 ** (len(data) * bitset_slot_size) - 1
    elif length == 1:
        singleton = frozenset(subsequence[0])
        bitset = generate_bitset(singleton, data,
                                 bitset_slot_size)
        itemsets_bitsets[singleton] = bitset
    else:
        # general case
        bitset = 2 ** (len(data) * bitset_slot_size) - 1
        first_iteration = True
        for itemset_i in subsequence:
            itemset = frozenset(itemset_i)

            try:
                itemset_bitset = itemsets_bitsets[itemset]
            except KeyError:
                # the bitset is not in the hashmap, we need to generate it
                itemset_bitset = generate_bitset(itemset, data,
                                                 bitset_slot_size)
                itemsets_bitsets[itemset] = itemset_bitset

            if first_iteration:
                first_iteration = False
            else:
                bitset = following_ones(bitset, bitset_slot_size,
                                        first_zero_mask)

                bitset &= itemset_bitset

    # now we just need to extract support, supersequence and class_pattern_count
    support = 0
    class_pattern_count = 0

    support, bitset_simple = get_support_from_vector(bitset,
                                                     bitset_slot_size,
                                                     first_zero_mask,
                                                     last_ones_mask)

    # find supersequences and count class pattern:
    i = bitset_simple.bit_length() - 1

    while i > 0:
        if bitset_simple >> i & 1:
            index_data = len(data) - i - 1

            if data[index_data][0] == target_class:
                class_pattern_count += 1

        i -= 1

    occurency_ratio = support / len(data)

    # we find the number of elements who have the right target_class
    class_pattern_ratio = class_pattern_count / support
    class_data_ratio = class_data_count / len(data)

    wracc = occurency_ratio * (class_pattern_ratio - class_data_ratio)

    return wracc


def count_subsequences_number(sequence):
    solutions = {0: 1}

    for i, x in enumerate(sequence):
        if x not in sequence[:i]:
            solutions[i+1] = 2 * solutions[i]
        else:
            last_index = 0
            for j, char in enumerate(sequence[i-1::-1]):
                if char == x:
                    last_index = j + 1
                    break

            solutions[i+1] = 2 * solutions[i] - solutions[last_index - 1]

    return solutions[len(sequence)]

def misere(data, time_budget, target_class, top_k=10):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    sorted_patterns = PrioritySet()

    bitset_slot_size = len(max(data, key=lambda x: len(x)))
    first_zero_mask = compute_first_zero_mask(len(data), bitset_slot_size)
    last_ones_mask = compute_last_ones_mask(len(data), bitset_slot_size)
    class_data_count = count_target_class_data(data, target_class)
    itemsets_bitsets = {}

    while datetime.datetime.utcnow() - begin < time_budget:
        sequence = copy.deepcopy(random.choice(data))
        sequence = sequence[1:]

        # for now we consider this upper bound (try better later)
        items = set([i for j_set in sequence for i in j_set])
        # ads = len(items) * (2 * len(sequence) - 1)
        ads = count_subsequences_number(sequence)

        for i in range(int(math.log(ads))):
            subsequence = copy.deepcopy(sequence)

            # we remove z items randomly
            seq_items_nb = len([i for j_set in subsequence for i in j_set])
            # print(seq_items_nb)
            z = random.randint(1, seq_items_nb - 1)

            for _ in range(z):
                chosen_itemset_i = random.randint(0, len(subsequence) - 1)
                chosen_itemset = subsequence[chosen_itemset_i]


                chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

                if len(chosen_itemset) == 0:
                    subsequence.pop(chosen_itemset_i)

            # now we calculate the Wracc
            # wracc = compute_WRAcc(data, subsequence, target_class)


            wracc = compute_WRAcc_vertical(data, subsequence, target_class,
                                           bitset_slot_size,
                                           itemsets_bitsets, class_data_count,
                                           first_zero_mask, last_ones_mask)

            sorted_patterns.add(sequence_mutable_to_immutable(subsequence),
                                wracc)

    print('Iterations misere: {}'.format(len(sorted_patterns.set)))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

'''
DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]
# DATA = read_data_kosarak('../data/all.csv')
results = misere(DATA, 10, '1')

print_results(results)
'''
