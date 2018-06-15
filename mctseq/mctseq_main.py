# -*- coding: utf-8 -*-

"""Main module."""
import datetime
import random
import cProfile
from pympler import classtracker

from mctseq.utils import read_data, read_data_kosarak, read_data_sc2, \
    extract_items, uct, \
    count_target_class_data, print_results_mcts, \
    subsequence_indices, sequence_immutable_to_mutable, \
    compute_first_zero_mask, compute_last_ones_mask, encode_items, encode_data, \
    decode_sequence, filter_redondant_result, create_graph
from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality



# Roll-out needs to be very quick !

class MCTSeq():
    def __init__(self, pattern_number, items, data, time_budget, target_class,
                 enable_i=True):
        self.pattern_number = pattern_number
        self.items = items

        self.time_budget = datetime.timedelta(seconds=time_budget)
        self.data = data
        self.target_class = target_class
        self.target_class_data_count = count_target_class_data(data,
                                                               target_class)
        self.enable_i = enable_i
        self.sorted_patterns = PrioritySetQuality()

        # the bitset_slot_size is the number max of itemset in all sequences
        self.bitset_slot_size = len(max(self.data, key=lambda x: len(x)))

        # this hashmap is a memory of bitsets of itemsets composing sequences.
        # {frozenset: bitset}
        self.itemsets_bitsets = {}
        self.first_zero_mask = compute_first_zero_mask(len(data),
                                                       self.bitset_slot_size)
        self.last_ones_mask = compute_last_ones_mask(len(data),
                                                     self.bitset_slot_size)
        self.root_node = SequenceNode([], None, self.items, self.data,
                                      self.target_class,
                                      self.target_class_data_count,
                                      self.itemsets_bitsets,
                                      self.enable_i,
                                      bitset_slot_size=self.bitset_slot_size,
                                      first_zero_mask=self.first_zero_mask,
                                      last_ones_mask=self.last_ones_mask)

        # contains sequence-SequenceNode for permutation-unification
        self.node_hashmap = {}

    def launch(self):
        """
        Launch the algorithm, specifying how many patterns we want to mine
        """
        begin = datetime.datetime.utcnow()

        self.node_hashmap[self.root_node.sequence] = self.root_node

        current_node = self.root_node

        iteration_count = 0

        while datetime.datetime.utcnow() - begin < self.time_budget:
            node_sel = self.select(current_node)

            if node_sel != None:
                node_expand = self.expand(node_sel)
                reward = self.roll_out(node_expand)
                self.update(node_expand, reward)
            else:
                # we enter here if we have a terminal node/dead_end.
                break
            iteration_count += 1

        # Now we need to explore the tree to get interesting subgroups
        # We use a priority queue to store elements, sorted by their quality
        self.explore_children(self.root_node, self.sorted_patterns)

        create_graph(self.root_node)
        print('Number iteration mcts: {}'.format(iteration_count))
        print('Number of nodes: {}'.format(len(self.sorted_patterns.set)))

        return self.sorted_patterns.get_top_k_non_redundant(
            self.pattern_number)

    def select(self, node):
        """
        Select the best node, using exploration-exploitation tradeoff
        :param node: the node from where we begin to search
        :return: the selected node, or None if exploration is finished
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            else:
                node = self.best_child(node)

                # If we reach this point, it means we reached a dead_end
                if node is None:
                    return None

        # if we reach this point, it means we reached a terminal node
        return None

    def expand(self, node):
        """
        Choose a child to expand among possible children of node
        :param node: the node from wich we want to expand
        :return: the expanded node
        """
        expanded_node = node.expand(self.node_hashmap)
        node.expand_children(self.node_hashmap)

        return expanded_node

    def roll_out(self, node):
        """
        Use the same idea as Misere (see paper)
        :param node: the node from wich launch the roll_out
        :return: the quality measure, depending on the reward agregation policy
        """
        # we have take iterate on supersequences (present in data) of sequenceNode
        # for each, we launch a number of generalisation, and we get the top-k
        # element

        best_patterns = PrioritySetQuality()

        if node.is_dead_end:
            return 0

        for sequence in node.dataset_sequences:
            # items = set([i for j_set in sequence for i in j_set])
            # ads = len(items) * (2 * len(sequence) - 1)

            # define here the number of generalisation we try.
            for i in range(3):
                # we remove z items randomly, if they are not in the intersection
                # between expanded_node and sursequences
                forbiden_itemsets = subsequence_indices(node.sequence,
                                                        sequence)

                seq_items_nb = len([i for j_set in sequence for i in j_set])
                z = random.randint(1, seq_items_nb)

                subsequence = sequence_immutable_to_mutable(sequence)

                for _ in range(z):
                    chosen_itemset_i = random.randint(0, len(subsequence) - 1)
                    chosen_itemset = subsequence[chosen_itemset_i]

                    # here we check if chosen_itemset is not a forbidden one.
                    if chosen_itemset_i not in forbiden_itemsets:
                        chosen_itemset.remove(
                            random.sample(chosen_itemset, 1)[0])

                        if len(chosen_itemset) == 0:
                            subsequence.pop(chosen_itemset_i)

                created_node = SequenceNode(subsequence, None,
                                            self.items, self.data,
                                            self.target_class,
                                            self.target_class_data_count,
                                            self.itemsets_bitsets,
                                            self.enable_i,
                                            bitset_slot_size=self.bitset_slot_size,
                                            first_zero_mask=self.first_zero_mask,
                                            last_ones_mask=self.last_ones_mask)

                best_patterns.add(created_node)

        # we take only the best element
        top_k_patterns = best_patterns.get_top_k(1)

        for i in top_k_patterns:
            self.sorted_patterns.add(i[1])

        # we can come to this case if we have a node wich is not present in data
        try:
            mean_quality = sum([i[0] for i in top_k_patterns]) / len(
                top_k_patterns)
        except ZeroDivisionError:
            mean_quality = 0

        return mean_quality

        # we return the best patter we found we this roll-out
        # return max(top_k_patterns, key=lambda x: x[0])[0]

    def update(self, node, reward):
        """
        Backtrack: update the node and recursively update all nodes until the root
        :param node: the node we want to update
        :param reward: the reward we got
        :return: None
        """
        node.update(reward)

        for parent in node.parents:
            if parent != None:
                self.update(parent, reward)

    def best_child(self, node):
        """
        Return the best child, based on UCT. Can only return a child which is
        not a dead_end
        :param node:
        :return: the best child, or None if we reached a dead_end
        """
        best_node = None
        max_score = -float("inf")


        for child in node.generated_children[:node.limit_generated_children]:
            current_uct = uct(node, child)
            if current_uct > max_score and not child.is_dead_end:
                max_score = current_uct
                best_node = child

        # special case: k first elements are dead end. We relax limit_generated_children
        if best_node is None and len(node.generated_children) > node.limit_generated_children:
            node.limit_generated_children = node.limit_generated_children + 1
            return self.best_child(node)

        #if best_node is None:
        #    print('cc')

        return best_node

    def explore_children(self, node, sorted_patterns):
        """
        Find children of node and add them to sorted_children
        :param node: the parent from which we explore children
        :param sorted_patterns: PrioritySetQuality.
        :return: None
        """
        for child in node.generated_children:
            sorted_patterns.add(child)
            self.explore_children(child, sorted_patterns)


if __name__ == '__main__':
    # DATA = read_data('../data/promoters.data')
    # DATA = read_data_kosarak('../data/out.data')
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]

    items = extract_items(DATA)

    items, item_to_encoding, encoding_to_item = encode_items(items)
    DATA = encode_data(DATA, item_to_encoding)

    mcts = MCTSeq(10, items, DATA, 50, '1', enable_i=False)

    #result = mcts.launch()
    #print_results_mcts(result, encoding_to_item)
    cProfile.run('mcts.launch()')
