import copy
import random

from mctseq.utils import sequence_immutable_to_mutable, \
    sequence_mutable_to_immutable, is_subsequence


class SequenceNode():
    def __init__(self, sequence, parent, candidate_items, data):
        # the pattern is in the form [{}, {}, ... ]
        self.sequence = sequence
        self.parent = parent
        self.number_visit = 1
        self.data = data
        self.candidate_items = candidate_items
        self.is_fully_expanded = False
        self.is_terminal = False
        self.quality = self.calculate_quality()
        self.support = self.calculate_support()

        # List of patterns
        self.non_generated_children = self.get_non_generated_children()

        # Set of generated children
        self.generated_children = set()

    def calculate_support(self):
        """
        Calculate the support of current element
        """
        # TODO: Optimize it (vertical representation, like in prefixspan ?)
        support = 0
        for row in self.data:
            if is_subsequence(self.sequence, row):
                support += 1

        self.support = support

    def calculate_quality(self):
        # TODO: update quality with WRAcc
        pass

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
        self.number_visit += 1
        self.quality = (self.number_visit * self.quality + reward) / (
                self.number_visit + 1)

    def expand(self):
        """
        Create a random children, and add it to generated children. Removes
        considered pattern from the possible_children
        :return: the SequenceNode created
        """
        index_pattern_children = random.randint(0, len(
            self.non_generated_children))

        pattern_children = self.non_generated_children.pop(
            index_pattern_children)

        self.update_node_state()

        expanded_node = SequenceNode(pattern_children, self,
                                     self.candidate_items, self.data)
        self.generated_children.add(expanded_node)

        return expanded_node

    def get_non_generated_children(self):
        """
        :return: the list of sequences that we can generate from the current one
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

                pseudo_i_extension = sequence_immutable_to_mutable(
                    copy.deepcopy(subsequence)
                )
                add = pseudo_i_extension[index].add(item)

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
