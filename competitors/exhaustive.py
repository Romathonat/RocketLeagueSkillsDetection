import datetime
import random
import copy
import math

from seqscout.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, compute_quality, compute_quality_vertical, create_s_extension, create_i_extension, extract_items,\
    reduce_k_length

from seqscout.priorityset import PrioritySet

def compute_children(sequence, items, enable_i=True):
    """
    :param enable_i: enable i_extensions or not. Useful when sequences are singletons like DNA
    :return: the set of sequences that we can generate from the current one
    NB: We convert to mutable/immutable object in order to have a set of subsequences,
    which automatically removes duplicates
    """
    new_subsequences = set()

    for item in items:
        for index, itemset in enumerate(sequence):
            new_subsequences.add(
                create_s_extension(sequence, item, index)
            )

            if enable_i:
                pseudo_i_extension = create_i_extension(sequence, item,
                                                        index)

                length_i_ext = sum([len(i) for i in pseudo_i_extension])
                len_subsequence = sum([len(i) for i in sequence])

                # we prevent the case where we add an existing element to itemset
                if (length_i_ext > len_subsequence):
                    new_subsequences.add(pseudo_i_extension)

        new_subsequences.add(
            create_s_extension(sequence, item, len(sequence)))

    return new_subsequences


def items_to_sequences(items):
    sequences = []
    for item in items:
        sequences.append((frozenset([item]),))

    return sequences


def display_info(stage, compute_count, sorted_patterns, begin, data, top_k):
    print("The algorithm is at stage {} and did {}".format(stage, compute_count))
    print("The algorithm took :{}".format(datetime.datetime.utcnow() - begin))
    print("We print the best patterns")
    patterns = sorted_patterns.get_top_k_non_redundant(data, top_k)
    print_results(patterns)

def exhaustive(data, target_class, top_k=5, enable_i=True):
    begin = datetime.datetime.utcnow()

    # by storing this large element, we avoid the problem of adding problems elements
    sorted_patterns = PrioritySet(500)

    bitset_slot_size = len(max(data, key=lambda x: len(x))) - 1
    first_zero_mask = compute_first_zero_mask(len(data), bitset_slot_size)
    last_ones_mask = compute_last_ones_mask(len(data), bitset_slot_size)
    class_data_count = count_target_class_data(data, target_class)
    itemsets_bitsets = {}

    items = extract_items(data)

    fifo = [[]]

    # to know if elements have already been added
    fifo_elements = set()

    stage = 0
    compute_count = 0

    while len(fifo) != 0:
        seed = fifo.pop(0)
        children = compute_children(seed, items, enable_i)

        if k_length(seed) > stage:
            stage = k_length(seed)
            display_info(stage, compute_count, sorted_patterns, begin, data, top_k)

        for child in children:
            quality, bitset = compute_quality_vertical(data, child, target_class,
                                                       bitset_slot_size,
                                                       itemsets_bitsets,
                                                       class_data_count,
                                                       first_zero_mask,
                                                       last_ones_mask)


            sorted_patterns.add_preserve_memory(child, quality, data)

            # we do not explore elements with a null support
            if child not in fifo_elements and bitset != 0:
                fifo.append(child)
                fifo_elements.add(child)

            compute_count += len(children)

    print("The algorithm took:{}".format(datetime.datetime.utcnow() - begin))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

def launch():
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    DATA = reduce_k_length(10, DATA)

    results = exhaustive(DATA, '1', top_k=10, enable_i=True)

    print_results(results)

if __name__ == '__main__':
   launch()
