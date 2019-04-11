from seqsamphill.utils import find_LCS, is_subsequence, sequence_mutable_to_immutable


class Node():
    def __init__(self, added_object, parent, data, target_class):
        '''
        :param added_object:
        :param extend: identifiers of objects
        :param parent:
        :param data:
        :param target_class:
        '''

        try:
            # case of the 1st floor
            if parent.intend == [frozenset({'.'})]:
                self.intend = added_object
            else:
                self.intend = sequence_mutable_to_immutable(find_LCS(parent.intend, added_object))
                if self.intend == ():
                    raise ValueError('No LCS')
        except AttributeError:
            # case of the root and rollout (not used)
            self.intend = [frozenset({'.'})]

        self.quality, self.extend = self.get_extend_and_quality(data, self.intend, target_class)

        if parent != None:
            self.parents = [parent]
            parent.children.append(self)
        else:
            self.parents = []

        self.children = []
        self.number_visits = 1

        # corresponds to the index of object to add to the node (when it's value is len(data) - 1, node is fully expanded
        self.i_expand = 0

    def get_normalized_wracc(self):
        return (self.quality + 0.25) * 2

    def get_extend_and_quality(self, data, subsequence, target_class):
        subsequence_supp = 0
        data_supp = len(data)
        class_subsequence_supp = 0
        class_data_supp = 0
        extend = []

        for i, sequence in enumerate(data):
            current_class = sequence[0]
            sequence = sequence[1:]

            if is_subsequence(subsequence, sequence):
                subsequence_supp += 1
                if current_class == target_class:
                    class_subsequence_supp += 1
                extend.append(i)
            if current_class == target_class:
                class_data_supp += 1

        if subsequence_supp == 0:
            return -float("inf"), extend

        return (subsequence_supp / data_supp) * (
            class_subsequence_supp / subsequence_supp -
            class_data_supp / data_supp), extend

    def update(self, reward):
        """
        Update the quality of the node
        :param reward: the roll-out score
        :return: None
        """
        # Mean-update
        self.quality = (self.number_visits * self.quality + reward) / (
            self.number_visits + 1)
        self.number_visits += 1
