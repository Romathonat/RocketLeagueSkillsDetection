from seqscout.utils import extract_items, extract_l_max, read_data_kosarak, read_data_sc2, read_data, read_jmlr
from math import factorial


# https://www.geeksforgeeks.org/generate-unique-partitions-of-an-integer/
def decompose(n):
    if n == 0:
        return [[]]
    decompositions = []
    p = [0] * n  # An array to store a partition
    k = 0  # Index of last element in a partition
    p[k] = n  # Initialize first partition
    # as number itself

    # This loop first prints current partition,
    # then generates next partition.The loop
    # stops when the current partition has all 1s
    while True:

        # print current partition
        decomposition = []
        for i in range(k + 1):
            decomposition.append(p[i])
        decompositions.append(decomposition)

        # Generate next partition

        # Find the rightmost non-one value in p[].
        # Also, update the rem_val so that we know
        # how much value can be accommodated
        rem_val = 0
        while k >= 0 and p[k] == 1:
            rem_val += p[k]
            k -= 1

        # if k < 0, all the values are 1 so
        # there are no more partitions
        if k < 0:
            return decompositions

        # Decrease the p[k] found above
        # and adjust the rem_val
        p[k] -= 1
        rem_val += 1

        # If rem_val is more, then the sorted
        # order is violated. Divide rem_val in
        # different values of size p[k] and copy
        # these values at different positions after p[k]
        while rem_val > p[k]:
            p[k + 1] = p[k]
            rem_val = rem_val - p[k]
            k += 1

        # Copy rem_val to next position
        # and increment position
        p[k + 1] = rem_val
        k += 1


def decomposition_histogram(decomposition):
    histo = {}
    for elt in decomposition:
        histo[elt] = histo.setdefault(elt, 0) + 1
    return histo


# print(decompose(1))
# print(decomposition_histogram(decompose(6)[3]))


def combination(k, n):
    if k > n:
        return 0

    return factorial(n) // (factorial(k) * factorial(n - k))


def compute_dataset_size(DATA):
    ITEMS = extract_items(DATA)
    m = len(ITEMS)
    stages = {}

    l_max = extract_l_max(DATA)

    pattern_number = 1  # we count the root

    for l in range(l_max + 1):
        stage_count = 0
        for k in range(l):
            decompositions = decompose(k)

            for decomposition in decompositions:
                # more set of balls to share than bags, impossible case
                if len(decomposition) <= l - k:
                    first_element = m ** (l - k - len(decomposition))

                    histo = decomposition_histogram(decomposition)
                    histo_factorial_product = 1
                    for _, unique_factor in histo.items():
                        histo_factorial_product *= factorial(unique_factor)

                    second_element = factorial(l - k) / (
                            factorial(l - k - len(decomposition)) * histo_factorial_product)

                    for elt in decomposition:
                        second_element *= combination(elt + 1, m)
                    stage_pattern = first_element * second_element
                    pattern_number += stage_pattern
                    stage_count += stage_pattern

            stages[l] = stage_count

    return pattern_number, stages


def compute_dataset_size_raissy(DATA):
    ITEMS = extract_items(DATA)
    m = len(ITEMS)

    l_max = extract_l_max(DATA)

    w_k = 0

    for i in range(50):
        local = combination(l_max, m * i) / (2 ** (i + 1))
        w_k += local

    return w_k


def w_k(memo, k, m):
    result = 0

    for i in range(k):
        if i in memo:
            w_i = memo[i]
        else:
            w_i = w_k(memo, i, m)
            memo[i] = w_i
        result += w_i * combination(k - i, m)

    return result


def compute_dataset_size_raissy2(DATA):
    ITEMS = extract_items(DATA)
    m = len(ITEMS)

    l_max = extract_l_max(DATA)

    memo = {0: 1}

    somme = w_k(memo, l_max, m)


    for i, value in memo.items():
        somme += value

    return somme


if __name__ == '__main__':
    datasets = [
        (read_data_kosarak('../data/aslbu.data'), '195', 'aslbu'),
        (read_data('../data/promoters.data'), '+', 'promoters'),
        (read_data_kosarak('../data/blocks.data'), '1', 'blocks'),
        (read_data_kosarak('../data/context.data'), '4', 'context'),
        (read_data('../data/splice.data'), 'EI', 'splice'),
        (read_data_sc2('../data/sequences-TZ-45.txt')[:5000], '1', 'sc2'),
        (read_data_kosarak('../data/skating.data'), '1', 'skating'),
        (read_jmlr('svm', '../data/jmlr/jmlr'), 'svm', 'jmlr')
    ]

    for dataset in datasets:
        DATA, class_target, dataset_name = dataset

        # we remove first element wich are useless
        for i in range(len(DATA)):
            DATA[i] = DATA[i][1:]

        #pattern_number, stages = compute_dataset_size(DATA)
        #print(pattern_number)
        # print(stages)

        pattern_number_2 = compute_dataset_size_raissy2(DATA)
        print(pattern_number_2)
