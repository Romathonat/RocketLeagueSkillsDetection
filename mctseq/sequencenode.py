import copy
import random

from mctseq.utils import sequence_immutable_to_mutable, \
    sequence_mutable_to_immutable, is_subsequence


class SequenceNode():
    def __init__(self, sequence, parent, candidate_items, data, target_class,
                 class_data_count, enable_i=True):
        # the pattern is in the form [{}, {}, ... ]
        # data is in the form [[class, {}, {}, ...], [class, {}, {}, ...]]

        self.sequence = sequence
        self.parent = parent
        self.number_visit = 1
        self.data = data
        self.candidate_items = candidate_items
        self.is_fully_expanded = False
        self.is_terminal = False
        self.target_class = target_class

        # those variables are here to compute WRacc
        self.class_pattern_count = 0
        self.class_data_count = class_data_count

        self.support = self.compute_support()
        self.quality = self.compute_quality()
        self.wracc = self.quality

        # set of patterns
        self.non_generated_children = self.get_non_generated_children(enable_i)

        # Set of generated children
        self.generated_children = set()

    def compute_support(self):
        """
        Compute the support of current element and class_pattern_count
        """
        # TODO: Optimize it (vertical representation, like in prefixspan ?)
        support = 0

        for row in self.data:
            if is_subsequence(self.sequence, row[1:]):
                support += 1
                if row[0] == self.target_class:
                    self.class_pattern_count += 1
        return support

    def compute_quality(self):
        # TODO: Maybe there is a better way to optimize this
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
        Update states is_terminal and is_fully_expanded
        """
        if len(self.non_generated_children) == 0:
            self.is_fully_expanded = True

            # if at least one children have support > 0, it is not a terminal node
            test_terminal = True
            for child in self.generated_children:
                if child.support > 0:
                    test_terminal = False

            self.is_terminal = test_terminal

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

    def expand(self):
        """
        Create a random children, and add it to generated children. Removes
        considered pattern from the possible_children
        :return: the SequenceNode created
        """

        pattern_children = random.sample(self.non_generated_children, 1)[0]

        self.non_generated_children.remove(pattern_children)


        expanded_node = SequenceNode(pattern_children, self,
                                     self.candidate_items, self.data,
                                     self.target_class, self.class_data_count)

        self.generated_children.add(expanded_node)
        self.update_node_state()

        return expanded_node

    def get_non_generated_children(self, enable_i=True):

        """
        :param enable_i: enable i_extensions or not. Useful when sequences are singletons like DNA
        :return: the set of sequences that we can generate from the current one
        NB: We convert to mutable/immutable object in order to have a set of subsequence,
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
                sequence_mutable_to_immutable(new_subsequence)
            )

        return new_subsequences