import datetime
import random
import copy
import pathlib

import math
import os

from seqsamphill.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, extract_items, compute_WRAcc, compute_WRAcc_vertical, jaccard_measure, find_LCS, \
    reduce_k_length, average_results, sequence_immutable_to_mutable

from seqsamphill.priorityset import PrioritySet, PrioritySetUCB

VERTICAL_TOOLS = {}
VERTICAL_RPZ = False


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


def is_included(pattern, pattern_set):
    if pattern in pattern_set:
        return True
    else:
        for x in pattern_set:
            if pattern.issubset(x):
                return True
        return False


def compute_variations_better_wracc(sequence, items, data, itemsets_memory, target_class, target_wracc, enable_i=True):
    '''
    Compute variations until quality increases
    :param sequence:
    :param items: the list of all possible items
    :return: the best new element (sequence, wracc), or None if we are on a local optimum
    '''
    variations = []

    if VERTICAL_RPZ:
        bitset_slot_size = VERTICAL_TOOLS['bitset_slot_size']
        itemsets_bitsets = VERTICAL_TOOLS['itemsets_bitsets']
        class_data_count = VERTICAL_TOOLS['class_data_count']
        first_zero_mask = VERTICAL_TOOLS['first_zero_mask']
        last_ones_mask = VERTICAL_TOOLS['last_ones_mask']

    for itemset_i, itemset in enumerate(sequence):
        # i_extension
        if enable_i:
            for item_possible in items:
                new_variation_i_extension = copy.deepcopy(sequence)
                new_variation_i_extension[itemset_i].add(item_possible)

                # we check if created pattern is present in data before
                if is_included(new_variation_i_extension, itemsets_memory):
                    if VERTICAL_RPZ:
                        new_variation_i_wracc, new_variation_i_bitset = compute_WRAcc_vertical(data,
                                                                                               new_variation_i_extension,
                                                                                               target_class,
                                                                                               bitset_slot_size,
                                                                                               itemsets_bitsets,
                                                                                               class_data_count,
                                                                                               first_zero_mask,
                                                                                               last_ones_mask)
                    else:
                        new_variation_i_wracc = compute_WRAcc(data, new_variation_i_extension, target_class)

                    variations.append(
                        (new_variation_i_extension, new_variation_i_wracc))

                    if new_variation_i_wracc > target_wracc:
                        return variations[-1]

        # s_extension
        for item_possible in items:
            new_variation_s_extension = copy.deepcopy(sequence)
            new_variation_s_extension.insert(itemset_i, {item_possible})

            if VERTICAL_RPZ:
                new_variation_s_wracc, new_variation_s_bitset = compute_WRAcc_vertical(data,
                                                                                       new_variation_s_extension,
                                                                                       target_class,
                                                                                       bitset_slot_size,
                                                                                       itemsets_bitsets,
                                                                                       class_data_count,
                                                                                       first_zero_mask,
                                                                                       last_ones_mask)
            else:
                new_variation_s_wracc = compute_WRAcc(data,
                                                      new_variation_s_extension,
                                                      target_class)

            variations.append(
                (new_variation_s_extension, new_variation_s_wracc))

            if new_variation_s_wracc > target_wracc:
                return variations[-1]

        for item_i, item in enumerate(itemset):
            new_variation_remove = copy.deepcopy(sequence)

            # we can switch this item, remove it or add it as s or i-extension

            if (k_length(sequence) > 1):
                new_variation_remove[itemset_i].remove(item)

                if len(new_variation_remove[itemset_i]) == 0:
                    new_variation_remove.pop(itemset_i)

                if VERTICAL_RPZ:
                    new_variation_remove_wracc, new_variation_remove_bitset = compute_WRAcc_vertical(data,
                                                                                                     new_variation_remove,
                                                                                                     target_class,
                                                                                                     bitset_slot_size,
                                                                                                     itemsets_bitsets,
                                                                                                     class_data_count,
                                                                                                     first_zero_mask,
                                                                                                     last_ones_mask)
                else:
                    new_variation_remove_wracc = compute_WRAcc(data,
                                                               new_variation_remove,
                                                               target_class)

                variations.append(
                    (new_variation_remove, new_variation_remove_wracc))
                if new_variation_remove_wracc > target_wracc:
                    return variations[-1]

    # s_extension for last element
    for item_possible in items:
        new_variation_s_extension = copy.deepcopy(sequence)
        new_variation_s_extension.append({item_possible})

        if VERTICAL_RPZ:
            new_variation_s_wracc, new_variation_s_bitset = compute_WRAcc_vertical(data,
                                                                                   new_variation_s_extension,
                                                                                   target_class,
                                                                                   bitset_slot_size,
                                                                                   itemsets_bitsets,
                                                                                   class_data_count,
                                                                                   first_zero_mask,
                                                                                   last_ones_mask)
        else:
            new_variation_s_wracc = compute_WRAcc(data,
                                                  new_variation_s_extension,
                                                  target_class)

        variations.append(
            (new_variation_s_extension, new_variation_s_wracc))
        if new_variation_s_wracc > target_wracc:
            return variations[-1]

    return None


def generalize_sequence(sequence, data, target_class):
    sequence = copy.deepcopy(sequence)
    # we remove z items randomly
    seq_items_nb = len([i for j_set in sequence for i in j_set])
    z = random.randint(0, seq_items_nb - 1)
    for _ in range(z):
        chosen_itemset_i = random.randint(0, len(sequence) - 1)
        chosen_itemset = sequence[chosen_itemset_i]

        chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

        if len(chosen_itemset) == 0:
            sequence.pop(chosen_itemset_i)

    # now we compute the Wracc
    if VERTICAL_RPZ:
        wracc, _ = compute_WRAcc_vertical(data, sequence, target_class,
                                          VERTICAL_TOOLS['bitset_slot_size'],
                                          VERTICAL_TOOLS['itemsets_bitsets'], VERTICAL_TOOLS['class_data_count'],
                                          VERTICAL_TOOLS['first_zero_mask'], VERTICAL_TOOLS['last_ones_mask'])
    else:
        wracc = compute_WRAcc(data, sequence, target_class)
    return sequence, wracc


def UCB(score, Ni, N):
    # * 2 to compensate the wracc ?
    return score + 2 * math.sqrt(math.log(N) / Ni)


def exploit_arm(pattern, wracc, items, data, itemsets_memory, target_class, enable_i=True):
    # we optimize until we find local optima
    while 'climbing hill':
        # we compute all possible variations
        try:

            pattern, wracc, _ = compute_variations_better_wracc(pattern,
                                                                items, data,
                                                                itemsets_memory,
                                                                target_class,
                                                                wracc,
                                                                enable_i=enable_i)

        except:
            break
    return pattern, wracc


def play_arm(sequence, data, target_class, min_quality_exploit, items, itemsets_memory, enable_i=True):
    '''
    Select object, generalise, use policy to exploit.
    Exploit Policy: Exploit if quality > min_top_k

    :param sequence: immutable sequence to generalise
    :param data:
    :param data_target_class: elements of the data with target class
    :param min_quality_exploit:
    :return:
    '''
    sequence = sequence_immutable_to_mutable(sequence)

    pattern, wracc = generalize_sequence(sequence,
                                         data,
                                         target_class)

    # OPTIMIZE: following policy
    if wracc > min_quality_exploit:
        exploit_arm(pattern, wracc, items, data, itemsets_memory, target_class, enable_i=enable_i)

    return pattern, wracc


def flat_UCB(data, items, time_budget, target_class, top_k=10, enable_i=True, vertical=True):
    # contains infos about elements of dataset. {sequence: (Ni, UCB, WRAcc)}. Must give the best UCB quickly. Priority queue
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    data_target_class = filter_target_class(data, target_class)
    sorted_patterns = PrioritySet(top_k)
    UCB_scores = PrioritySetUCB()
    itemsets_memory = get_itemset_memory(data)

    # removing class
    bitset_slot_size = len(max(data, key=lambda x: len(x))) - 1

    global VERTICAL_RPZ
    VERTICAL_RPZ = vertical

    global VERTICAL_TOOLS
    VERTICAL_TOOLS = {
        "bitset_slot_size": bitset_slot_size,
        "first_zero_mask": compute_first_zero_mask(len(data), bitset_slot_size),
        "last_ones_mask": compute_last_ones_mask(len(data), bitset_slot_size),
        "class_data_count": count_target_class_data(data, target_class),
        "itemsets_bitsets": {}
    }

    N = 1

    # init
    # we generalise only elements with target class
    for sequence in data_target_class:
        sequence_i = sequence_mutable_to_immutable(sequence[1:])
        # we try to only do misere at the beginning
        pattern, wracc = play_arm(sequence_i, data, target_class, 0.25, items, itemsets_memory, enable_i=enable_i)
        sorted_patterns.add(sequence_i, wracc)

        UCB_score = UCB(wracc, 1, N)
        UCB_scores.add(sequence_i, (UCB_score, 1, wracc))

        N += 1

    # play with time budget
    while datetime.datetime.utcnow() - begin < time_budget:
        # we take the best UCB
        _, Ni, mean_wracc, sequence = UCB_scores.pop()

        # we get the last score of top_k: we will exploit if we have more
        # min_score = sorted_patterns.get_top_k_non_redundant(data, top_k)[-1]

        # we exploit if score is better than 50% of the best
        min_score = sorted_patterns.get_top_k(1)[0][0] / 2

        pattern, wracc = play_arm(sequence, data, target_class, min_score, items, itemsets_memory, enable_i=enable_i)
        pattern = sequence_mutable_to_immutable(pattern)
        sorted_patterns.add(pattern, wracc)

        # we update scores
        updated_wracc = (Ni * mean_wracc + wracc) / (Ni + 1)
        UCB_score = UCB(updated_wracc, Ni + 1, N)
        UCB_scores.add(sequence, (UCB_score, Ni + 1, updated_wracc))

        N += 1

    print("Flat UCB iterations: {}".format(N))
    # print(UCB_scores.heap)
    return sorted_patterns.get_top_k_non_redundant(data, top_k)


def launch():
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]
    # DATA = read_mushroom()

    DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')
    ITEMS = extract_items(DATA)

    results = flat_UCB(DATA, ITEMS, 120, '+', top_k=10, enable_i=False, vertical=True)
    print_results(results)


# TODO: memory preservation
if __name__ == '__main__':
    launch()
