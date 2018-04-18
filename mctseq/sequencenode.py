import copy
import random

from bitarray import bitarray

from mctseq.utils import sequence_immutable_to_mutable, \
    sequence_mutable_to_immutable, is_subsequence, immutable_seq, k_length, \
    generate_bitset, following_ones


class SequenceNode():
    def __init__(self, sequence, parent, candidate_items, data, target_class,
                 class_data_count, bitset_slot_size, itemsets_bitsets,
                 enable_i=True):
        # the pattern is in the form [{}, {}, ... ]
        # data is in the form [[class, {}, {}, ...], [class, {}, {}, ...]]

        self.sequence = immutable_seq(sequence)
        if parent != None:
            self.parents = [parent]
        else:
            self.parents = []

        self.number_visit = 0
        self.data = data
        self.candidate_items = candidate_items
        self.is_fully_expanded = False
        self.is_terminal = False
        self.enable_i = enable_i
        self.bitset_slot_size = bitset_slot_size

        # a node is a dead end if is terminal, or if all its children are dead_end too
        # It means that is is useless to explore it, because it lead to terminal children
        self.is_dead_end = False

        self.target_class = target_class

        # those variables are here to compute WRacc
        self.class_pattern_count = 0
        self.class_data_count = class_data_count

        # dataset_sequence contains one super-sequence present in the dataset
        (self.support, self.dataset_sequence, self.class_pattern_count,
         self.bitset) = self.compute_support(itemsets_bitsets)

        self.quality = self.compute_quality()
        self.wracc = self.quality

        # set of patterns
        self.non_generated_children = self.get_non_generated_children(enable_i)

        # Set of generated children
        self.generated_children = set()

        # we update node state, in case supp = 0 it is a dead end
        self.update_node_state()

    def __eq__(self, other):
        try:
            return self.sequence == other.sequence
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self.sequence)

    def __lt__(self, other):
        return self.quality < other.quality

    def __gt__(self, other):
        return self.quality > other.quality

    def __str__(self):
        return '{}'.format(self.sequence)

    def __repr__(self):
        return '{}'.format(self.sequence)

    def compute_support(self, itemsets_bitsets):
        """
        :param itemsets_bitsets: the hashmap of biteset of itemsets known
        Compute the support of current element and class_pattern_count.
        :return:  support, sursequence, class_pattern_sequence, and bitset, a n
        number with its bit reprensentation representing data
        """
        # we have two cases: if length is <= 1, or not
        length = k_length(self.sequence)
        bitset = 0
        if length == 0:
            # the empty node is present everywhere
            # we just have to create a vector of ones
            bitset = 2 ** (len(self.data) * self.bitset_slot_size) - 1
            support = len(self.data)
            supersequence = self.data[random.randint(0, len(self.data) - 1)]
            class_pattern_count = self.class_data_count

        elif length == 1:
            bitset = self.generate_bitset(self.sequence[0])
        else:
            # general case
            final_bitset = 2 ** (len(self.data) * self.bitset_slot_size) - 1
            for itemset in self.sequence:
                try:
                    itemset_bitset = itemsets_bitsets[itemset]
                except KeyError:
                    # the bitset is not in the hashmap, we need to generate it
                    # an optimisation would be to find a subset of size k-1, and to make a
                    # & with item bitset

                    itemset_bitset = generate_bitset(itemset)
                    itemsets_bitsets[itemset] = itemset_bitset

                final_bitset = following_ones(final_bitset)
                final_bitset &= itemsets_bitsets

        # now we just need to extract support, supersequence and class_pattern_count

        support = 0
        class_pattern_count = 0
        supersequence = None

        for i in range(final_bitset.bit_length(), 0, -1):
            if final_bitset >> i & 1:
                support += 1

                index_data = (
                    (final_bitset.bit_length() - i) /
                    self.bitset_slot_size)

                if self.data[index_data][0] == self.target_class:
                    class_pattern_count += 1

                # this means that the last super_sequence is taken (may induce a strong bias)
                supersequence = self.data[index_data]

                i += self.bitset_slot_size - (i % self.bitset_slot_size) + 1

        return support, supersequence, class_pattern_count, bitset


def compute_quality(self):
    try:
        occurency_ratio = self.support / len(self.data)

        # we find the number of elements who have the right target_class
        class_pattern_ratio = self.class_pattern_count / self.support
        class_data_ratio = self.class_data_count / len(self.data)

        return occurency_ratio * (class_pattern_ratio - class_data_ratio)
    except ZeroDivisionError:
        return 0


def update_node_state(self):
    """
    Update states is_terminal, is_fully_expanded and is_dead_end
    """
    # a node cannot be terminal if one or more of its children are not expanded
    if len(self.non_generated_children) == 0:
        self.is_fully_expanded = True

        # if at least one children have support > 0, it is not a terminal node
        test_terminal = True
        test_dead_end = True
        for child in self.generated_children:
            if child.support > 0:
                test_terminal = False

            if not child.is_dead_end:
                test_dead_end = False

        self.is_terminal = test_terminal
        self.is_dead_end = test_dead_end

        # now we need to recursively update parents of current child, to
        # update if they are dead_end or not
        if self.is_dead_end:
            for parent in self.parents:
                parent.update_node_state()

    # in all cases, if support is null it is a dead end
    if self.support == 0:
        self.is_dead_end = True


def update(self, reward):
    """
    Update the quality of the node
    :param reward: the roll-out score
    :return: None
    """
    # Mean-update
    self.quality = (self.number_visit * self.quality + reward) / (
        self.number_visit + 1)
    self.number_visit += 1


def expand(self, node_hashmap):
    """
    Create a random children, and add it to generated children. Removes
    considered pattern from the possible_children
    :param node_hashmap: the hashmap of MCTS nodes
    :return: the SequenceNode created
    """

    pattern_children = random.sample(self.non_generated_children, 1)[0]

    self.non_generated_children.remove(pattern_children)

    if pattern_children in node_hashmap:
        expanded_node = node_hashmap[pattern_children]
        expanded_node.parents.append(self)
    else:
        expanded_node = SequenceNode(pattern_children, self,
                                     self.candidate_items, self.data,
                                     self.target_class,
                                     self.class_data_count,
                                     self.bitset_slot_size,
                                     self.enable_i)

        node_hashmap[pattern_children] = expanded_node

    self.generated_children.add(expanded_node)
    self.update_node_state()

    return expanded_node


def get_non_generated_children(self, enable_i=True):
    """
    :param enable_i: enable i_extensions or not. Useful when sequences are singletons like DNA
    :return: the set of sequences that we can generate from the current one
    NB: We convert to mutable/immutable object in order to have a set of subsequences,
    which automatically removes duplicates
    """
    new_subsequences = set()
    subsequence = self.sequence

    for item in self.candidate_items:
        for index, itemset in enumerate(subsequence):
            s_extension = sequence_immutable_to_mutable(
                copy.deepcopy(subsequence)
            )

            s_extension.insert(index, {item})

            new_subsequences.add(
                sequence_mutable_to_immutable(s_extension)
            )

            if enable_i:
                pseudo_i_extension = sequence_immutable_to_mutable(
                    copy.deepcopy(subsequence)
                )

                pseudo_i_extension[index].add(item)

                length_i_ext = sum([len(i) for i in pseudo_i_extension])
                len_subsequence = sum([len(i) for i in subsequence])

                # we prevent the case where we add an existing element to itemset
                if (length_i_ext > len_subsequence):
                    new_subsequences.add(
                        sequence_mutable_to_immutable(pseudo_i_extension)
                    )

        new_subsequence = sequence_immutable_to_mutable(
            copy.deepcopy(subsequence)
        )

        new_subsequence.insert(len(new_subsequence), {item})

        new_subsequences.add(
            sequence_mutable_to_immutable(new_subsequence))

    return new_subsequences
