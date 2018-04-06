# -*- coding: utf-8 -*-

"""Main module."""
import datetime
import random

from mctseq.utils import read_data, extract_items, uct, \
    count_target_class_data, sequence_mutable_to_immutable
from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality


# TODO: stop when exaustive search has been made
# TODO: filter redondant elements (post process)
# TODO: Normalize Wracc !!!

# TODO: use bit set to keep extend -> see SPADE too know how to make a temporal join,
# and optimize a lot
# TODO: optimize is_subsequence (not necessary if we do the previous step)

# TODO: add permutation unification
# TODO: implement misere with Wracc
# TODO: better rollout strategies

### LATER
# TODO: Suplementary material notebook
# TODO: Visualisation graph


# IDEA: instead of beginning with the null sequence, take the sequence of max
# length, and enumerate its subsequences (future paper)

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

        # contains sequence-SequenceNode for permutation-unification
        self.root_node = SequenceNode([], None, self.items, self.data,
                                      self.target_class,
                                      self.target_class_data_count,
                                      self.enable_i)

        self.node_hashmap = {}

    def launch(self):
        """
        Launch the algorithm, specifying how many patterns we want to mine
        :return:
        """
        begin = datetime.datetime.utcnow()

        # current_node is root

        # TODO: simplify by transforming [] to ()
        root_key = sequence_mutable_to_immutable(self.root_node.sequence)
        self.node_hashmap[root_key] = self.root_node

        current_node = self.root_node

        while datetime.datetime.utcnow() - begin < self.time_budget:
            node_sel = self.select(current_node)
            node_expand = self.expand(node_sel)

            # TODO:give argument for max length
            reward = self.roll_out(node_expand, 5)
            self.update(node_expand, reward)

        # Now we need to explore the tree to get interesting subgroups
        # We use a priority queue to store elements, sorted by their quality

        self.explore_children(self.root_node, self.sorted_patterns)

        return self.sorted_patterns.get_top_k(self.pattern_number)

    def select(self, node):
        """
        Select the best node, using exploration-exploitation tradeoff
        :param node: the node from where we begin to search
        :return: the selected node, or node if exploration is finished
        """
        while not node.is_dead_end:
            if not node.is_fully_expanded:
                return node
            else:
                node = self.best_child(node)

        # if we reach this point, it means the tree is finished
        return node

    def expand(self, node):
        """
        Choose a child to expand among possible children of node
        :param node: the node from wich we want to expand
        :return: the expanded node
        """
        return node.expand(self.node_hashmap)

    def roll_out(self, node, max_length):
        """
        Equivalent to simulation in classical MCTS
        :param node: the node from wich launch the roll_out
        :param max_length: the number of refinements we make
        :return: the quality measure, depending on the reward agregation policy
        """
        # naive-roll-out
        # max-reward
        max_quality = node.quality
        top_node = node

        for i in range(max_length):
            pattern_child = random.sample(node.non_generated_children, 1)[0]

            # we create successively all node, without remembering them (easy coding for now)
            node = SequenceNode(pattern_child, node, self.items, self.data,
                                self.target_class,
                                self.target_class_data_count,
                                self.enable_i)

            if max_quality < node.quality:
                max_quality = node.quality
                top_node = node

        # we add the top node to sorted patterns
        self.sorted_patterns.add(top_node)

        return max_quality

    def update(self, node, reward):
        """
        Backtrack: update the node and recursively update all nodes until the root
        :param node: the node we want to update
        :param reward: the reward we got
        :return: None
        """
        # mean-update
        node.update(reward)

        for parent in node.parents:
            if parent != None:
                self.update(parent, reward)

    def best_child(self, node):
        """
        Return the best child, based on UCT. Can only return a child which is
        not a dead_end
        :param node:
        :return: the best child
        """
        best_node = None
        max_score = -float("inf")
        for child in node.generated_children:
            if uct(node, child) > max_score and not child.is_dead_end:
                max_score = child.quality
                best_node = child

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


# TODO: command line interface, with pathfile of data, number of patterns and max_time

if __name__ == '__main__':
    ITEMS = set()
    #DATA = read_data('../data/promoters.data')
    DATA = [['+', {'A'}, {'B'}]]

    # TODO: clean those data
    items = extract_items(DATA)

    mcts = MCTSeq(5, items, DATA, 50, '+', False)
    print(mcts.launch())
