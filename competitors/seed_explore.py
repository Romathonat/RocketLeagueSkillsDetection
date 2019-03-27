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
    reduce_k_length, average_results

from seqsamphill.priorityset import PrioritySet, THETA

from data.read_mushroom import read_mushroom

VERTICAL_TOOLS = {}
VERTICAL_RPZ = False


def is_included(pattern, pattern_set):
    if pattern in pattern_set:
        return True
    else:
        for x in pattern_set:
            if pattern.issubset(x):
                return True
        return False


def compute_variations_better_wracc(sequence, items, data, itemsets_memory, target_class, target_wracc, enable_i=False):
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


def filter_target_class(data, target_class):
    filter_data = []
    for line in data:
        if line[0] == target_class:
            filter_data.append(line)

    return filter_data


def extract_best_elements_path(path, theta):
    '''
    :param path: path in the form of (WRAcc, sequence, bitset)
    :param theta: similarity
    :return:
    '''
    try:
        best_wracc, best_sequence, best_bitset = path[-1]
    except:
        return []

    best_sequences = [(best_wracc, best_sequence)]

    # need to compare with each previous added element
    for wracc, sequence, bitset in reversed(path):
        if VERTICAL_RPZ:
            if jaccard_measure(bitset, best_bitset, VERTICAL_TOOLS['bitset_slot_size'],
                               VERTICAL_TOOLS['first_zero_mask'], VERTICAL_TOOLS['last_ones_mask']) < theta:
                best_sequences.append((wracc, sequence))
                best_sequence, best_bitset = sequence, bitset
        else:
            best_sequences.append((wracc, sequence))
            best_sequences, best_bitset = sequence, bitset

    return best_sequences


def create_seed(data, target_class, data_target_class):
    # sample
    sequence = copy.deepcopy(random.choice(data_target_class))
    sequence = sequence[1:]

    seed, quality = generalize_sequence(sequence,
                                        data,
                                        target_class)
    return (seed, quality)


def UCB(x, t, ni):
    '''
    :param t: iteration number
    :param ni: number of times current arm has been playedk
    :return:
    '''
    return x + math.sqrt((2 * math.log2(t)) / ni)


def get_score(x, t, ni, improvement, best_score):
    '''
    :param x: the score of current pattern
    :param t:
    :param ni:
    :param improvement: the % of improvement for last value
    :param best_score: best pattern found in the data so far
    :return: a score between 0 and 1
    '''
    # the score is based on the delta of quality increase, and on the difference to the best pattern found so far.
    # we consider positive elements so we shift the wracc
    best_score += 0.25
    x += 0.25
    #return min(0.5, improvement * 0.5) + (x - best_score) / best_score
    #return min(0.5, improvement * 0.5)
    return improvement


def select_arm(seeds, iterations_count, best_wracc):
    best_seed = ()
    best_score = -float('inf')

    for original_seed, (mean, ti, variation, improvement) in seeds.items():
        score_compute = get_score(mean, ti, iterations_count, improvement, best_wracc)
        if score_compute >= best_score:
            best_score = score_compute
            best_seed = original_seed

    return best_seed


def get_itemset_memory(data):
    memory = set()
    for line in data:
        for itemset in line[1:]:
            memory.add(frozenset(itemset))
    return memory


def seed_explore(data, items, time_budget, target_class, top_k=10, enable_i=True, vertical=True):
    # TODO: normalize quality !
    # TODO: improve memory strategy
    # first term exploitation, second exploration: if last increase is weak, give less points. If all last elements are weak, remove element.
    # multi arms bandit, infinite arms, increasing reward. WRAcc normalize with better estimation than 0.25 !

    begin = datetime.datetime.utcnow()
    time_budget = datetime.timedelta(seconds=time_budget)

    data_target_class = filter_target_class(data, target_class)
    sorted_patterns = PrioritySet(top_k)
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

    iterations_count = 1
    optima_nb = 0

    # {original_seed: (quality, ti, improved_pattern, improvement)} diff with preceding value
    seeds = {}

    while datetime.datetime.utcnow() - begin < time_budget:
        # we keep a pool of k-elements we are looking for
        if len(seeds) < top_k:
            seed, quality = create_seed(data, target_class, data_target_class)

            seed_immu = sequence_mutable_to_immutable(seed)
            seeds[seed_immu] = (quality, 1, seed, 0)
            sorted_patterns.add(seed_immu, quality)

        else:
            best_origin_seed = select_arm(seeds, iterations_count, sorted_patterns.get_top_k(1)[0][0])
            quality, ti, best_seed, diff_quality = seeds[best_origin_seed]

            try:
                improved_best_seed, best_quality = compute_variations_better_wracc(best_seed,
                                                                                   items, data,
                                                                                   itemsets_memory,
                                                                                   target_class,
                                                                                   quality,
                                                                                   enable_i=enable_i)

                sorted_patterns.add(sequence_mutable_to_immutable(improved_best_seed), best_quality)

                # we shift the quality to divide by a number > 0
                improvement = best_quality + 0.25 - (quality + 0.25) / (quality + 0.25)
                seeds[best_origin_seed] = (
                    best_quality, ti + 1, improved_best_seed, best_quality - quality)

            except TypeError:
                # we found a local optima
                # print('Found optima !')
                optima_nb += 1
                del seeds[best_origin_seed]

        iterations_count += 1

    print('Seed_Explore iterations:{}, {} optima and {} seeds'.format(iterations_count, optima_nb, len(seeds)))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)


def launch():
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    # DATA = read_mushroom()

    # DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')
    ITEMS = extract_items(DATA)

    results = seed_explore(DATA, ITEMS, 12, '1', top_k=10, enable_i=True, vertical=False)
    print_results(results)


if __name__ == '__main__':
    launch()

# TODO: for scoring: give points if delta is good, and points if we are far away from the best solution we found so far.
# or normalize it: between 0 and 0.5 for delta, and 0 and 0.5 for score (normalizing wracc
