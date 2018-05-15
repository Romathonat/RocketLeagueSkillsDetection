import random
import copy

from mctseq.utils import k_length

SEQUENCE_NB = 1000
K_LENGTH = 15
ITEM_NB = 5
ITEMS = set(range(5))
ITEMSET_SIZE = (1, 5)
PATTERN_SIZE = (1, 12)
PATTERN_NUMBER = 10
PATTERN_POS_PROBA = 0.8


def generate_pattern():
    pattern_length = random.randint(PATTERN_SIZE[0], PATTERN_SIZE[1])
    pattern = []

    while k_length(pattern) < pattern_length:
        itemset_size = random.randint(ITEMSET_SIZE[0], ITEMSET_SIZE[1])
        itemset = set(random.sample(ITEMS, itemset_size))
        pattern.append(itemset)

    return pattern


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
        if random.random() > 0.5:
            s_extension = set(random.sample(ITEMS, random.randint(ITEMSET_SIZE[0],
                                                              ITEMSET_SIZE[1])))

            if k_length(new_pattern) + len(s_extension) >= K_LENGTH:
                break

            new_pattern.append(s_extension)

    # we need to translate to kosarak here
    output = ''

    output = kosarak_translate(new_pattern)

    # adding label
    output += ' + \n'

    return output


def generate_noise():
    k_length_random = random.randint(1, K_LENGTH)
    pattern = []

    while k_length(pattern) < k_length_random:
        itemset = random.sample(ITEMS, random.randint(ITEMSET_SIZE[0],
                                                      ITEMSET_SIZE[1]))
        pattern.append(itemset)

    output = kosarak_translate(pattern)

    if random.random() > 0.5:
        output += '- \n'
    else:
        output += '+ \n'

    return output


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

# generating patterns
for pattern_i in range(PATTERN_NUMBER):
    pattern = generate_pattern()
    patterns.append(pattern)

    # we add patterns in data
    pattern_occurency = random.randint(0.02 * SEQUENCE_NB, 0.05 * SEQUENCE_NB)

    for _ in range(pattern_occurency):
        noisy_pattern = generate_noisy_pattern(pattern)

        output += noisy_pattern
        line_number += 1

# generate noise
for noise_i in range(line_number, SEQUENCE_NB):
    output += generate_noise()

with open('patterns', 'w') as f:
    patterns_out = ''

    for pattern in patterns:
        pattern = kosarak_translate(pattern)
        pattern += ' \n'
        patterns_out += pattern

    f.write(patterns_out)

with open('out.data', 'w') as f:
    f.write(output)
