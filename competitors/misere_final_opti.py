import datetime
import random
import copy
import math
import pathlib
import cProfile

from seqsamphill.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, compute_WRAcc, compute_WRAcc_vertical, reduce_k_length, sequence_immutable_to_mutable, \
    extract_items

from seqsamphill.priorityset import PrioritySet

def compute_variations_better_wracc(sequence, items, data, target_class,
                                    bitset_slot_size,
                                    itemsets_bitsets, class_data_count,
                                    first_zero_mask,
                                    last_ones_mask, target_wracc,
                                    enable_i=False, wracc_vertical=True):
    '''
    Compute variations until quality increases
    :param sequence:
    :param items: the list of all possible items
    :return: the best new element (sequence, wracc), or None if we are on a local optimum
    '''
    variations = []

    for itemset_i, itemset in enumerate(sequence):
        # i_extension
        if enable_i:
            for item_possible in items:
                new_variation_i_extension = copy.deepcopy(sequence)
                new_variation_i_extension[itemset_i].add(item_possible)

                if wracc_vertical:
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
                    new_variation_i_bitset = 0

                variations.append(
                    (new_variation_i_extension, new_variation_i_wracc, new_variation_i_bitset))

                if new_variation_i_wracc > target_wracc:
                    return variations[-1]

        # s_extension
        for item_possible in items:
            new_variation_s_extension = copy.deepcopy(sequence)
            new_variation_s_extension.insert(itemset_i, {item_possible})

            if wracc_vertical:
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
                new_variation_s_bitset = 0

            variations.append(
                (new_variation_s_extension, new_variation_s_wracc, new_variation_s_bitset))

            if new_variation_s_wracc > target_wracc:
                return variations[-1]

        for item_i, item in enumerate(itemset):
            new_variation_remove = copy.deepcopy(sequence)

            # we can switch this item, remove it or add it as s or i-extension

            if (k_length(sequence) > 1):
                new_variation_remove[itemset_i].remove(item)

                if len(new_variation_remove[itemset_i]) == 0:
                    new_variation_remove.pop(itemset_i)

                if wracc_vertical:
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
                    new_variation_remove_bitset = 0

                variations.append(
                    (new_variation_remove, new_variation_remove_wracc, new_variation_remove_bitset))
                if new_variation_remove_wracc > target_wracc:
                    return variations[-1]

    # s_extension for last element
    for item_possible in items:
        new_variation_s_extension = copy.deepcopy(sequence)
        new_variation_s_extension.append({item_possible})

        if wracc_vertical:
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
            new_variation_s_bitset = 0

        variations.append(
            (new_variation_s_extension, new_variation_s_wracc, new_variation_s_bitset))
        if new_variation_s_wracc > target_wracc:
            return variations[-1]

    return None


def exploit_pattern(pattern, wracc, items, data, target_class, bitset_slot_size, itemsets_bitsets, class_data_count,
                    first_zero_mask, last_ones_mask, enable_i=True):
    # we optimize until we find local optima
    # print("Optimize")
    while 'climbing hill':
        # we compute all possible variations
        try:

            pattern, wracc, _ = compute_variations_better_wracc(pattern,
                                                             items, data,
                                                             target_class,
                                                             bitset_slot_size,
                                                             itemsets_bitsets,
                                                             class_data_count,
                                                             first_zero_mask,
                                                             last_ones_mask,
                                                             wracc,
                                                             enable_i=enable_i)

        except TypeError:
            # print("Already a local optima")
            break
    return pattern, wracc


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


def misere_final_opti(data, ITEMS, time_budget, target_class, top_k=5):
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
    best_patterns = sorted_patterns.get_top_k_non_redundant(data, top_k)

    for pattern in best_patterns:
        pattern_mutable = sequence_immutable_to_mutable(pattern[1])
        optimized_pattern, optimized_wracc = exploit_pattern(pattern_mutable, pattern[0], ITEMS, data, target_class,
                                                             bitset_slot_size, itemsets_bitsets, class_data_count,
                                                             first_zero_mask, last_ones_mask, enable_i=True)

        optimized_pattern = sequence_mutable_to_immutable(optimized_pattern)
        sorted_patterns.add(optimized_pattern, optimized_wracc)

    print("Misere opti iterations:{}".format(iterations_count))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)


def launch():
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    # DATA = read_data_kosarak('../data/debile.data')

    DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')

    ITEMS = extract_items(DATA)

    # DATA = reduce_k_length(50, DATA)

    results = misere_final_opti(DATA, ITEMS, 12, '1', top_k=10)

    print_results(results)

if __name__ == '__main__':
    launch()
    #cProfile.runctx('launch()', globals(), locals())
