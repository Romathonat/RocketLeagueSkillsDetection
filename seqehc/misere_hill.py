import datetime
import random
import copy
import math

from seqehc.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, extract_items

from seqehc.priorityset import PrioritySet


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
    try:
        class_pattern_ratio = class_pattern_count / support
    except ZeroDivisionError:
        return -0.25
    class_data_ratio = class_data_count / len(data)

    wracc = occurency_ratio * (class_pattern_ratio - class_data_ratio)

    return wracc


def compute_variations(sequence, items, data, target_class, bitset_slot_size,
                       itemsets_bitsets, class_data_count, first_zero_mask,
                       last_ones_mask):
    '''
    Comput all variations with one step, with the WRAcc
    :param sequence:
    :param items: the list of all possible items
    :return: the variations, with their wracc in the form [(sequence, wracc), (..., ...), ...]
    '''
    variations = []

    for itemset_i, itemset in enumerate(sequence):
        for item_i, item in enumerate(itemset):
            new_variation_remove = copy.deepcopy(sequence)

            # we can switch this item, or remove it
            # TODO: add i and s-extension
            if (k_length(sequence) > 1):
                new_variation_remove[itemset_i].remove(item)

                if len(new_variation_remove[itemset_i]) == 0:
                    new_variation_remove.pop(itemset_i)

                new_variation_remove_wracc = compute_WRAcc_vertical(data,
                                                                    new_variation_remove,
                                                                    target_class,
                                                                    bitset_slot_size,
                                                                    itemsets_bitsets,
                                                                    class_data_count,
                                                                    first_zero_mask,
                                                                    last_ones_mask)

                variations.append(
                    (new_variation_remove, new_variation_remove_wracc))

            for item_possible in items:
                new_variation = copy.deepcopy(sequence)
                new_variation[itemset_i].remove(item)
                new_variation[itemset_i].add(item_possible)
                new_variation_wracc = compute_WRAcc_vertical(data,
                                                             new_variation,
                                                             target_class,
                                                             bitset_slot_size,
                                                             itemsets_bitsets,
                                                             class_data_count,
                                                             first_zero_mask,
                                                             last_ones_mask)

                variations.append((new_variation, new_variation_wracc))

    return variations


def generalize_sequence(sequence, data, target_class, bitset_slot_size,
                        itemsets_bitsets, class_data_count, first_zero_mask,
                        last_ones_mask):
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
    wracc = compute_WRAcc_vertical(data, sequence, target_class,
                                   bitset_slot_size,
                                   itemsets_bitsets, class_data_count,
                                   first_zero_mask, last_ones_mask)
    return (sequence, wracc)


def misere_hill(data, items, time_budget, target_class, top_k=10):
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

        current_sequence, current_wracc = generalize_sequence(sequence, data,
                                                              target_class,
                                                              bitset_slot_size,
                                                              itemsets_bitsets,
                                                              class_data_count,
                                                              first_zero_mask,
                                                              last_ones_mask)

        while 'climbing hill':
            # we compute all possible variations
            variations = compute_variations(current_sequence, items, data,
                                            target_class,
                                            bitset_slot_size,
                                            itemsets_bitsets,
                                            class_data_count,
                                            first_zero_mask,
                                            last_ones_mask)

            # we take the best solution, and we iterate
            sequence, wracc = max(variations, key=lambda x: x[1])

            if wracc > current_wracc:
                current_sequence = sequence
                current_wracc = wracc
            else:
                break

        sorted_patterns.add(sequence_mutable_to_immutable(current_sequence),
                            wracc)

    print('Iterations misere hill: {}'.format(len(sorted_patterns.set)))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

'''
DATA = read_data_sc2('./data/sequences-TZ-45.txt')[:100]
ITEMS = extract_items(DATA)
# DATA = read_data_kosarak('../data/all.csv')
results = misere_hill(DATA, ITEMS, 5, '1')
print_results(results)
'''
