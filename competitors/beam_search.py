import datetime
import pathlib

from seqsamphill.priorityset import PrioritySet
from seqsamphill.utils import count_target_class_data, compute_last_ones_mask, \
    compute_first_zero_mask, create_s_extension, sequence_immutable_to_mutable, \
    create_i_extension, k_length, generate_bitset, following_ones, \
    get_support_from_vector, read_data_sc2, extract_items, print_results, compute_WRAcc, compute_WRAcc_vertical, read_data_kosarak, read_data


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


def beam_search(data, items, time_budget, target_class, enable_i=True,
                top_k=5, beam_width=50):
    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    bitset_slot_size = len(max(data, key=lambda x: len(x))) - 1

    first_zero_mask = compute_first_zero_mask(len(data), bitset_slot_size)
    last_ones_mask = compute_last_ones_mask(len(data), bitset_slot_size)
    class_data_count = count_target_class_data(data, target_class)
    itemsets_bitsets = {}

    # candidate_queue = items_to_sequences(items)
    candidate_queue = [[]]

    sorted_patterns = PrioritySet(top_k)


    while datetime.datetime.utcnow() - begin < time_budget:
        beam = PrioritySet()

        while (len(candidate_queue) != 0):
            seed = candidate_queue.pop(0)
            children = compute_children(seed, items, enable_i)

            for child in children:
                quality, _ = compute_WRAcc_vertical(data, child, target_class,
                                                 bitset_slot_size,
                                                 itemsets_bitsets,
                                                 class_data_count,
                                                 first_zero_mask,
                                                 last_ones_mask)

                #sorted_patterns.add_preserve_memory(child, quality, data)
                sorted_patterns.add(child, quality)
                beam.add(child, quality)

        candidate_queue = [j for i, j in beam.get_top_k_non_redundant(data, beam_width)]
        #candidate_queue = [j for i, j in beam.get_top_k(beam_width)]

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

def launch():
    #DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]
    DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')

    #DATA = read_data_kosarak('../data/debile.data')
    items = extract_items(DATA)

    results = beam_search(DATA, items, 180, '+', enable_i=False, top_k=10, beam_width=30)
    print_results(results)


if __name__ == '__main__':
   launch()
