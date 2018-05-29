import random
import copy

from mctseq.utils import k_length, compute_WRAcc

SEQUENCE_NB = 50
K_LENGTH = 500

# ITEM_NB = 5

ITEMS = set(range(10))
ITEMSET_SIZE = (1, 10)
PATTERN_SIZE = (5, 8)
PATTERN_NUMBER = 2
PATTERN_PROPORTION = 0.1


def generate_pattern():
    pattern_length = random.randint(PATTERN_SIZE[0], PATTERN_SIZE[1])
    pattern = []

    while k_length(pattern) < pattern_length:
        itemset_size = random.randint(ITEMSET_SIZE[0], ITEMSET_SIZE[1])
        itemset = set(random.sample(ITEMS, itemset_size))
        pattern.append(itemset)

    return pattern


def is_superset(itemset, itemsets):
    """
    :param itemsets: set of all itemsets
    :return: True if itemset is a superset of a least on element of itemsets,
    else False
    """
    for itemset_i in itemsets:
        if itemset_i.issubset(itemset):
            return True

    return False


def generate_noisy_pattern(pattern):
    new_pattern = []

    for itemset in pattern:
        absent_items = ITEMS - itemset
        new_itemset = copy.deepcopy(itemset)

        # adding doing i-extension
        for _ in range(len(absent_items)):
            # arbitrary
            min_element_picked = min(len(absent_items) - 1, 2)
            adding_items_nb = random.randint(0, min_element_picked)
            new_itemset = new_itemset.union(
                set(random.sample(absent_items, adding_items_nb)))

        if k_length(new_pattern) + len(new_itemset) >= K_LENGTH:
            break

        new_pattern.append(new_itemset)

        # adding doing s-extension
        if random.random() > 0.3:
            s_extension = set(
                random.sample(ITEMS, random.randint(ITEMSET_SIZE[0],
                                                    ITEMSET_SIZE[1])))

            if k_length(new_pattern) + len(s_extension) >= K_LENGTH:
                break

            new_pattern.append(s_extension)

    # we need to translate to kosarak here
    output = ''

    output = kosarak_translate(new_pattern)

    # adding label
    output += ' + \n'
    new_pattern.insert(0, '+')

    return output, new_pattern


def extract_itemsets_pattern(patterns):
    itemsets = set()
    for pattern in patterns:
        for itemset in pattern:
            itemsets.add(frozenset(itemset))

    return itemsets


def generate_noise(patterns):
    """
    We add a constraint: the sequence must not contain one of the pattern initially generated
    :return: noise sequence
    """
    k_length_random = random.randint(1, K_LENGTH)
    pattern = []
    forbidden_itemsets = extract_itemsets_pattern(patterns)
    while k_length(pattern) < k_length_random:
        itemset = set(random.sample(ITEMS, random.randint(ITEMSET_SIZE[0],
                                                          ITEMSET_SIZE[1])))
        # simple approach: we forbid the use of supersets of itemsets present in generated pattern
        if not is_superset(itemset, forbidden_itemsets):
            pattern.append(itemset)

    output = kosarak_translate(pattern)

    if random.random() > 0.00005:
        output += '- \n'
        pattern.insert(0, '-')
    else:
        output += '+ \n'
        pattern.insert(0, '+')

    return output, pattern


def kosarak_translate(pattern):
    output = ''
    for itemset in pattern:
        itemset_out = ''
        for item in itemset:
            itemset_out += '{} '.format(item)

        itemset_out += '-1 '
        output += itemset_out

    output += '-2 '
    return output


output = ''
line_number = 0
patterns = []
data = []

# generating patterns
for pattern_i in range(PATTERN_NUMBER):
    pattern = generate_pattern()
    patterns.append(pattern)

    # we add patterns in data
    pattern_occurency = int(
        PATTERN_PROPORTION * SEQUENCE_NB / PATTERN_NUMBER)

    for _ in range(pattern_occurency):
        noisy_pattern, noisy_pattern_python = generate_noisy_pattern(pattern)

        while k_length(noisy_pattern_python) < K_LENGTH:
            noisy_pattern, noisy_pattern_python = generate_noisy_pattern(
                noisy_pattern_python[1:])

        data.append(noisy_pattern_python)

        output += noisy_pattern
        line_number += 1

# generate noise
for noise_i in range(line_number, SEQUENCE_NB):
    noise_string, noise_python = generate_noise(patterns)
    output += noise_string
    data.append(noise_python)

with open('patterns', 'w') as f:
    patterns_out = ''
    for pattern in patterns:
        wracc = compute_WRAcc(data, pattern, '+')
        pattern = kosarak_translate(pattern)
        pattern += ' {} \n'.format(wracc)
        patterns_out += pattern

    f.write(patterns_out)

with open('../data/out.data', 'w') as f:
    f.write(output)
