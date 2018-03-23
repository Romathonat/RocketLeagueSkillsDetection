import copy

from mctseq.utils import sequence_immutable_to_mutable, \
    sequence_mutable_to_immutable


class SequenceNode():
    def __init__(self, sequence, parent, candidate_items):
        # the pattern is in the form [{}, {}, ... ]
        self.sequence = sequence
        self.parent = parent
        self.quality = 0
        self.number_visit = 0
        self.is_fully_expanded = False

        # useless ?
        # self.items

        self.possible_children = self.get_possible_children(candidate_items)

        self.generated_children = set()

    def get_possible_children(self, candidate_items):
        """
        :return: the list of sequences that we can generate from the current one
        NB: We convert to mutable/immutable object in order to have a set of subsequence,
        which automatically removes duplicates
        """
        new_subsequences = set()
        subsequence = self.sequence

        for item in candidate_items:
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
