import datetime
import random
import copy
import math

from mctseq.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results

from mctseq.priorityset import PrioritySet


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


def misere(data, time_budget, target_class, top_k=10):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    sorted_patterns = PrioritySet()

    while datetime.datetime.utcnow() - begin < time_budget:
        sequence = copy.deepcopy(random.choice(data))
        sequence = sequence[1:]

        # for now we consider this upper bound (try better later)
        items = set([i for j_set in sequence for i in j_set])
        ads = len(items) * (2 * len(sequence) - 1)

        for i in range(int(math.log(ads))):
            subsequence = copy.deepcopy(sequence)

            # we remove z items randomly
            seq_items_nb = len([i for j_set in subsequence for i in j_set])
            # print(seq_items_nb)
            z = random.randint(1, seq_items_nb - 1)

            for _ in range(z):
                chosen_itemset_i = random.randint(0, len(subsequence) - 1)
                chosen_itemset = subsequence[chosen_itemset_i]

                #print(subsequence)
                #print(chosen_itemset)

                chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

                if len(chosen_itemset) == 0:
                    subsequence.pop(chosen_itemset_i)

            # now we calculate the Wracc

            wracc = compute_WRAcc(data, subsequence, target_class)

            sorted_patterns.add(sequence_mutable_to_immutable(subsequence),
                                wracc)

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

'''
DATA = read_data('../data/promoters.data')
#DATA = read_data_kosarak('../data/all.csv')
results = misere(DATA, 5, '+')

print_results(results)
'''
