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
            sequence = []
            sequence.append(line[-2])
            line = line[:-2]

            line_split = line.split("-1")[:-1]

            for itemset in line_split:
                items = itemset.split()
                new_itemset = set(items)
                sequence.append(new_itemset)

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


def jaccard_measure(node1, node2):
    intersec = hamming_weight(node1.bitset_simple & node2.bitset_simple)
    union = hamming_weight(node1.bitset_simple | node2.bitset_simple)

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

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    return bitset


def get_support_from_vector(bitset, bitset_slot_size, first_zero_mask,
                            last_ones_mask):

    bitset = bitset >> 1
    bitset = bitset & first_zero_mask

    temp = bitset

    for i in range(bitset_slot_size - 1):
        temp = temp >> 1
        temp = temp & first_zero_mask
        bitset |= temp

    bitset = bitset & last_ones_mask

    # now we have a vector with ones or 0 at the end of each slot. We just need to
    # compute the hamming distance
    return hamming_weight(bitset)


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


def print_results(results):
    for result in results:
        pattern_display = ''

        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))


def print_results_mcts(results, encoding_to_items):
    for result in results:
        pattern_display = ''

        sequence = decode_sequence(result[1].sequence, encoding_to_items)
        for itemset in sequence:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))


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
        f.write(graphviz_string)


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
