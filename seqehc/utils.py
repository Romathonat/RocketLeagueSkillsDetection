import math


def sequence_mutable_to_immutable(sequence):
    """
    :param sequence: form [{}, {}, ...]
    :return: the same sequence in its immutable form
    """
    return tuple([frozenset(i) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    """
    :param sequence: form (frozenset(), frozenset(), ...)
    :return: the same sequence in its mutable form
    """
    return [set(i) for i in sequence]


def create_s_extension(sequence, item, index):
    """
    Perform an s-extension
    :param sequence: the sequence we are extending
    :param item: the item to insert (not a set, an item !)
    :param index: the index to add the item
    :return: an immutable sequence
    """
    # .insert would require a deep copy, wich is not performance

    new_sequence = []
    appended = False

    for i, itemset in enumerate(sequence):
        if i == index:
            new_sequence.append(frozenset({item}))
            appended = True
        new_sequence.append(itemset)

    if not appended:
        new_sequence.append(frozenset({item}))

    return tuple(new_sequence)


def create_i_extension(sequence, item, index):
    """
    Perform an i-extension
    :param sequence: the sequence we are extending
    :param item: the item to merge to(not a set, an item !)
    :param index: the index to add the item
    :return: an immutable sequence
    """
    new_sequence = []

    for i, itemset in enumerate(sequence):
        if i == index:
            new_sequence.append(frozenset({item}).union(itemset))
        else:
            new_sequence.append(itemset)

    return tuple(new_sequence)


def immutable_seq(sequence):
    """
    :param sequence: a seq wich is mutable or not
    :return: an immutable seq
    """
    if type(sequence) == list:
        return sequence_mutable_to_immutable(sequence)
    else:
        return sequence


def count_target_class_data(data, target_class):
    """
    Count the number of occurences of target_class in the data
    :param data: sequential data of for [[class, {}, {} ...], [class, {}, {}], ...]
    :param target_class: the targeted class
    :return: the count
    """
    count = 0
    for row in data:
        if row[0] == target_class:
            count += 1

    return count


def k_length(sequence):
    """
    :param sequence: the considered sequence
    :return: the length of the sequence
    """
    return sum([len(i) for i in sequence])


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


def is_subsequence(a, b):
    """ check if sequence a is a subsequence of b
    """
    index_b_mem = 0

    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            if index_a < len(a) - 1:
                # we reach the end of b and there are still elements in a
                return False
            elif itemset_a.issubset(b[-1]):
                # we reach the end of a and b, a_last_elt is included in
                # b_last_elt
                return True
            else:
                # we reach the end of a and b, a_last_elt is not included in
                # b_last_elt
                return False

    return True


def subsequence_indices(a, b):
    """ Return itemset indices of b that itemset of a are included in
        Precondition: a is a subset of b
    """
    index_b_mem = 0
    indices_b = []
    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                indices_b.append(index_b)
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            return indices_b

    return indices_b


def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            line_split = line.split(',')
            sequence = [line_split[0]]
            sequence += line_split[2].strip()

            sequences.append(sequence)

    for sequence in sequences:
        for itemset_i in range(1, len(sequence)):
            sequence[itemset_i] = set(sequence[itemset_i])

    return sequences


def read_data_kosarak(filename):
    """
    :param filename:
    :return: [[class, {}, {}, ...], [class, {}, {}, ...]]
    """
    data = []
    with open(filename) as f:
        for line in f:
            line_split = line.split("-1")
            first, second = line_split[0].split()
            line_split = line_split[1:-1]

            line_split.insert(0, second)

            sequence = [first]

            for itemset in line_split:
                items = itemset.split()
                new_itemset = set(items)
                sequence.append(new_itemset)

            data.append(sequence)
    return data


def read_data_sc2(filename):
    """
    :param filename:
    :return: [[class, {}, {}, ...], [class, {}, {}, ...]]
    """
    data = []
    with open(filename) as f:
        for line in f:
            sequence = []
            sequence.append(line[-8])
            line = line[:-8]

            line_split = line.split("-1")[:-2]

            for itemset in line_split:
                items = itemset.split()
                new_itemset = set(items)
                sequence.append(new_itemset)

            if len(sequence) > 1:
                data.append(sequence)
    return data


def encode_data(data, item_to_encoding):
    """
    Replaces all item in data by its encoding
    :param data:
    :param item_to_encoding:
    :return:
    """
    for line in data:
        for i, itemset in enumerate(line[1:]):
            encoded_itemset = set()
            for item in itemset:
                encoded_itemset.add(item_to_encoding[item])
            line[i + 1] = encoded_itemset

    return data


def decode_sequence(sequence, encoding_to_item):
    """
    Give the true values of sequence
    :param sequence: the sequence to decode in the form [{}, ..., {}]
    :return: the decoded sequence
    """
    return_sequence = []

    for i, itemset in enumerate(sequence):
        decoded_itemset = set()
        for item in itemset:
            decoded_itemset.add(encoding_to_item[item])
        return_sequence.append(decoded_itemset)
    return return_sequence


def encode_items(items):
    item_to_encoding = {}
    encoding_to_item = {}
    new_items = set()

    for i, item in enumerate(items):
        item_to_encoding[item] = i
        encoding_to_item[i] = item
        new_items.add(i)

    return new_items, item_to_encoding, encoding_to_item


def hamming_weight(vector):
    w = 0
    while vector:
        w += 1
        vector &= vector - 1
    return w


def jaccard_measure(bitset1, bitset2, bitset_slot_size, first_zero_mask, last_ones_mask):

    _, bitset1 = get_support_from_vector(bitset1, bitset_slot_size, first_zero_mask, last_ones_mask)
    _, bitset2 = get_support_from_vector(bitset2, bitset_slot_size, first_zero_mask, last_ones_mask)

    intersec = hamming_weight(bitset1 & bitset2)
    union = hamming_weight(bitset1 | bitset2)

    try:
        return intersec / union
    except ZeroDivisionError:
        return 0


def extract_items(data):
    """
    :param data: date must be on the form [[class, {}, {}, ...], [class, {}, {}, ...]]
    :return: set of items extracted
    """
    items = set()
    for sequence in data:
        for itemset in sequence[1:]:
            for item in itemset:
                items.add(item)
    return items


def uct(node, child_node):
    return child_node.quality + (2 / math.sqrt(2)) * math.sqrt(
        (2 * math.log(node.number_visit)) / child_node.number_visit)


def compute_first_zero_mask(data_length, bitset_slot_size):
    first_zero = 2 ** (bitset_slot_size - 1) - 1
    first_zero_mask = 0

    for i in range(data_length):
        first_zero_mask |= first_zero << i * bitset_slot_size

    return first_zero_mask


def compute_last_ones_mask(data_length, bitset_slot_size):
    last_ones = 1
    last_ones_mask = 1

    for i in range(data_length):
        last_ones_mask |= last_ones << i * bitset_slot_size

    return last_ones_mask


def following_ones(bitset, bitset_slot_size, first_zero_mask):
    """
    Transform bitset with 1s following for each 1 encoutered, for
    each bitset_slot.
    :param bitset:
    :param bitset_slot_size: the size of a slot in the bitset
    :return: a bitset (number)
    """
    # the first one needs to be a zero
    bitset = bitset >> 1
    bitset = bitset & first_zero_mask

    temp = bitset >> 1
    temp = temp & first_zero_mask

    bitset |= temp

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    return bitset


def get_support_from_vector(bitset, bitset_slot_size, first_zero_mask,
                            last_ones_mask):
    temp = bitset >> 1
    temp = temp & first_zero_mask

    bitset |= temp

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    bitset = bitset & last_ones_mask

    i = bitset.bit_length()

    data_length = math.ceil(i / bitset_slot_size)

    bitset_simple = 0
    count = 0

    while i > 0:
        if bitset >> (i - 1) & 1:
            bitset_simple |= 1 << (data_length - count - 1)

        count += 1
        i -= bitset_slot_size

    # now we have a vector with ones or 0 at the end of each slot. We just need to
    # compute the hamming distance
    return hamming_weight(bitset_simple), bitset_simple


def generate_bitset(itemset, data, bitset_slot_size):
    """
    Generate the bitset of itemset

    :param itemset: the itemset we want to get the bitset
    :param data: the dataset
    :return: the bitset of itemset
    """
    bitset = 0

    # we compute the extend by scanning the database
    for line in data:
        line = line[1:]
        sequence_bitset = 0
        for itemset_line in line:
            if itemset.issubset(itemset_line):
                bit = 1
            else:
                bit = 0

            sequence_bitset |= bit
            sequence_bitset = sequence_bitset << 1

        # for last element we need to reshift
        sequence_bitset = sequence_bitset >> 1

        # we shift to complete with 0
        sequence_bitset = sequence_bitset << bitset_slot_size - (len(line))

        # we add this bit vector to bitset
        bitset |= sequence_bitset
        bitset = bitset << bitset_slot_size

    # for the last element we need to reshift
    bitset = bitset >> bitset_slot_size

    return bitset


def compute_bitset_slot_size(data):
    max_size_itemset = 1

    for line in data:
        max_size_line = len(max(line, key=lambda x: len(x)))
        if max_size_line > max_size_itemset:
            max_size_itemset = max_size_line

    return max_size_itemset


def print_results(results):
    sum_result = 0
    for result in results:
        pattern_display = ''
        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        sum_result += result[0]

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))

    print('Average score :{}'.format(sum_result / len(results)))


def print_results_mcts(results, encoding_to_items):
    sum_result = 0
    for result in results:
        pattern_display = ''

        sequence = decode_sequence(result[1].sequence, encoding_to_items)
        for itemset in sequence:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))
        sum_result += result[0]

    print('Average score :{}'.format(sum_result / len(results)))


def average_results(results):
    sum_result = 0
    for result in results:
        sum_result += result[0]

    return sum_result / len(results)


def format_sequence_graph(sequence):
    sequence_string = ''
    for itemset in sequence:
        itemset_string = ''

        for item in itemset:
            itemset_string += str(item)

        itemset_string = '{{{}}}, '.format(itemset_string)
        sequence_string += itemset_string

    sequence_string = '<{}>'.format(sequence_string[:-1])
    return sequence_string



# Require Graphviz
# Launch command:
# dot -Tpng graph.gv -o MCTSgraph.png
def create_graph(root_node):
    sequences = {}

    explore_graph(root_node, root_node, sequences, set())

    k_number = max(sequences)
    k_string = ''

    for i in range(k_number + 1):
        k_string += '{} -> '.format(i)
    k_string = k_string[:-4]

    graph_construction = ''
    edges_construction = ''

    for key, level_sequences in sequences.items():
        level_string = ''

        for level_sequence in level_sequences:
            level_string += '"{}"; '.format(
                level_sequence[0])
            edges_construction += '{}'.format(level_sequence[1])

        level_string = '{{ rank = same; {}; {} }} \n'.format(key, level_string)
        graph_construction += level_string

    graphviz_string = """
    digraph MCTSGraph {{
        {{
            {}; 
        }} 
        node[label=""];
        {} 
        {}
    }}
    """.format(k_string, graph_construction, edges_construction)

    with open('../graph.gv', 'w+') as f:
        # f.write(graphviz_string)
        f.write(k_string)


def explore_graph(node, parent, sequences, seen):
    k = k_length(node.sequence)
    sequences.setdefault(k, []).append(
        (node.sequence,
         '"{}" -> "{}"; \n'.format(parent.sequence, node.sequence)))

    # we add child only if this node has not been seen before
    if node.sequence not in seen:
        for child in node.generated_children:
            explore_graph(child, node, sequences, seen)
        seen.add(node.sequence)


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

    try:
        wracc = (subsequence_supp / data_supp) * (
            class_subsequence_supp / subsequence_supp -
            class_data_supp / data_supp)

        return wracc

    except:
        return 0


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

                # aie aie aie !
                bitset = itemset_bitset
            else:
                bitset = following_ones(bitset, bitset_slot_size,
                                        first_zero_mask)

                bitset &= itemset_bitset

    # now we just need to extract support, supersequence and class_pattern_count
    class_pattern_count = 0

    support, bitset_simple = get_support_from_vector(bitset,
                                                     bitset_slot_size,
                                                     first_zero_mask,
                                                     last_ones_mask)

    # find supersequences and count class pattern:
    i = bitset_simple.bit_length() - 1

    while i >= 0:
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
        return -0.25, 0
    class_data_ratio = class_data_count / len(data)

    wracc = occurency_ratio * (class_pattern_ratio - class_data_ratio)

    return wracc, bitset

def backtrack_LCS(C, seq1, seq2, i, j, lcs):
    if i == 0 or j == 0:
        return

    inter = seq1[i-1].intersection(seq2[j-1])

    if inter != set():
        lcs.insert(len(lcs) - 1, inter)
        return backtrack_LCS(C, seq1, seq2, i-1, j-1, lcs)
    if C[i][j-1] > C[i-1][j]:
        return backtrack_LCS(C, seq1, seq2, i, j-1, lcs)
    else:
        return backtrack_LCS(C, seq1, seq2, i-1, j, lcs)

def find_LCS(seq1, seq2):
    """
    find the longest common subsequence. We here consider sequences of itemsets
    :param seq1:
    :param seq2:
    :return: the longest common sequence

    We have to adapt the LCS algorithm to sequences of itemsets: instead of testing the
    """
    C = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            inter = seq1[i-1].intersection(seq2[j-1])
            if inter != set():
                C[i][j] = C[i-1][j-1] + len(inter)
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1])

    # now we need to backtrack the structure to get the pattern

    lcs = []
    backtrack_LCS(C, seq1, seq2, len(seq1), len(seq2), lcs)

    return lcs

