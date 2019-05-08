import datetime
import random
import copy
import math
import pathlib

from seqscout.utils import read_data, read_data_kosarak, \
    sequence_mutable_to_immutable, print_results, \
    read_data_sc2, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, compute_quality_vertical, read_jmlr

from seqscout.priorityset import PrioritySet
import seqscout.conf as conf


def count_subsequences_number(sequence):
    solutions = {0: 1}

    for i, x in enumerate(sequence):
        if x not in sequence[:i]:
            solutions[i + 1] = 2 * solutions[i]
        else:
            last_index = 0
            for j, char in enumerate(sequence[i - 1::-1]):
                if char == x:
                    last_index = j + 1
                    break

            solutions[i + 1] = 2 * solutions[i] - solutions[last_index - 1]

    return solutions[len(sequence)]


def misere(data, target_class, time_budget=conf.TIME_BUDGET, top_k=conf.TOP_K, iterations_limit=conf.ITERATIONS_NUMBER,
           theta=conf.THETA, quality_measure=conf.QUALITY_MEASURE):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    sorted_patterns = PrioritySet(theta=theta)

    bitset_slot_size = len(max(data, key=lambda x: len(x))) - 1
    first_zero_mask = compute_first_zero_mask(len(data), bitset_slot_size)
    last_ones_mask = compute_last_ones_mask(len(data), bitset_slot_size)
    class_data_count = count_target_class_data(data, target_class)
    itemsets_bitsets = {}

    iterations_count = 0

    while datetime.datetime.utcnow() - begin < time_budget and iterations_count < iterations_limit:
        sequence = copy.deepcopy(random.choice(data))
        sequence = sequence[1:]

        ads = count_subsequences_number(sequence)

        for i in range(int(math.log(ads))):
            if iterations_count >= iterations_limit:
                break

            subsequence = copy.deepcopy(sequence)

            # we remove z items randomly
            seq_items_nb = len([i for j_set in subsequence for i in j_set])
            z = random.randint(1, seq_items_nb - 1)

            for _ in range(z):
                chosen_itemset_i = random.randint(0, len(subsequence) - 1)
                chosen_itemset = subsequence[chosen_itemset_i]

                chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

                if len(chosen_itemset) == 0:
                    subsequence.pop(chosen_itemset_i)

            quality, _ = compute_quality_vertical(data, subsequence, target_class,
                                                bitset_slot_size,
                                                itemsets_bitsets, class_data_count,
                                                first_zero_mask, last_ones_mask, quality_measure=quality_measure)

            iterations_count += 1
            sorted_patterns.add(sequence_mutable_to_immutable(subsequence),
                                quality)

    return sorted_patterns.get_top_k_non_redundant(data, top_k)


def launch():
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    # DATA = read_data_kosarak('../data/blocks.data')
    # DATA = read_data_kosarak('../data/skating.data')

    DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')
    # DATA = read_data_kosarak('../data/aslbu.data')
    # DATA = read_jmlr('machin', pathlib.Path(__file__).parent.parent / 'data/jmlr/jmlr')

    results = misere(DATA, '+', time_budget=2 ** 30, iterations_limit=10, quality_measure='WRAcc')

    print_results(results)


if __name__ == '__main__':
    launch()
