import copy


class SequenceNode():
    def __init__(self, sequence, parent, candidate_items):
        # the pattern is in the form [{}, {}, ... ]
        self.sequence = sequence
        self.parent = parent
        self.quality = 0
        self.number_visit = 0
        self.is_fully_expanded = False

        # useless ?
        #self.items


        self.possible_children = self.get_possible_children(candidate_items)

        self.generated_children = set()

    def get_possible_children(self, candidate_items):
        """
        :return: the list of sequences that we can generate from the current one
        """
        new_subsequences = list()
        subsequence = self.pattern

        for item in candidate_items:
            for index, itemset in enumerate(subsequence):
                s_extension = copy.deepcopy(subsequence)
                s_extension.insert(index, {item})
                new_subsequences.append(s_extension)

                pseudo_i_extension = copy.deepcopy(subsequence)
                add = pseudo_i_extension[index].add(item)

                length_i_ext = sum([len(i) for i in pseudo_i_extension])
                len_subsequence = sum([len(i) for i in subsequence])

                # we prevent the case where we add an existing element to itemset
                if (length_i_ext > len_subsequence):
                    new_subsequences.append(pseudo_i_extension)

            new_subsequence = copy.deepcopy(subsequence)
            new_subsequence.insert(len(new_subsequence), {item})
            new_subsequences.append(new_subsequence)

        return new_subsequences
