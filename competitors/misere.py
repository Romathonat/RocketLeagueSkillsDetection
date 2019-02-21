import datetime
import random
import copy
import math

from seqehc.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, compute_WRAcc, compute_WRAcc_vertical, reduce_k_length

from seqehc.priorityset import PrioritySet


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

def misere(data, time_budget, target_class, top_k=5):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    sorted_patterns = PrioritySet()

    bitset_slot_size = len(max(data, key=lambda x: len(x))) - 1
    first_zero_mask = compute_first_zero_mask(len(data), bitset_slot_size)
    last_ones_mask = compute_last_ones_mask(len(data), bitset_slot_size)
    class_data_count = count_target_class_data(data, target_class)
    itemsets_bitsets = {}

    iterations_count = 0

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

            wracc, _ = compute_WRAcc_vertical(data, subsequence, target_class,
                                           bitset_slot_size,
                                           itemsets_bitsets, class_data_count,
                                           first_zero_mask, last_ones_mask)

            iterations_count += 1
            sorted_patterns.add(sequence_mutable_to_immutable(subsequence),
                                wracc)

    print("Misere iterations:{}".format(iterations_count))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

def launch():
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    #DATA = read_data_kosarak('../data/debile.data')
    # DATA = read_data_kosarak('../data/all.csv')
    DATA = read_data('../data/promoters.data')

    DATA = reduce_k_length(50, DATA)
    results = misere(DATA, 10, '+', top_k=10)

    print_results(results)

if __name__ == '__main__':
   launch()
