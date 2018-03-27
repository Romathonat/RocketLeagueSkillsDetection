# -*- coding: utf-8 -*-

"""Main module."""
import datetime
import random

from mctseq.utils import read_data, extract_items, uct
from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality


class MCTSeq():
    def __init__(self, pattern_number, items, data, time_budget, target_class):
        self.pattern_number = pattern_number
        self.items = items
        self.time_budget = time_budget
        self.data = data
        self.target_class = target_class

    def launch(self):
        """
        Launch the algorithm, specifying how many patterns we want to mine
        :return:
        """
        begin = datetime.datetime.utcnow()

        # current_node is root
        root_node = SequenceNode([], None, items, self.data, self.target_class)
        current_node = root_node

        while datetime.datetime.utcnow() - begin < self.time_budget:
            node_sel = self.select(current_node)
            node_expand = self.expand(node_sel)
            reward = self.roll_out(node_expand)
            self.update(node_expand, reward)

        # Now we need to explore the tree to get interesting subgroups
        # We use a priority queue to store elements, sorted by their quality
        sorted_patterns = PrioritySetQuality()

        self.explore_children(root_node, sorted_patterns)

        return sorted_patterns.get_top_k(self.pattern_number)

    def select(self, node):
        """
        Select the best node, using exploration-exploitation tradeoff
        :param node: the node from where we begin to search
        :return: the selected node
        """
        # TODO: What do we do in the case where the node is terminal (we don't want to generate children)
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        """
        Choose a child to expand among possible children of node
        :param node: the node from wich we want to expand
        :return: the expanded node
        """
        return node.expand()

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

        for i in range(max_length):
            index_pattern_child = random.randint(0,
                                                 node.non_generated_children)
            pattern_child = node.non_generated_children[index_pattern_child]

            # we create successively all node, without remembering them (easy coding for now)
            node = SequenceNode(pattern_child, node, self.items, self.data,
                                node.target_class)
            max_quality = max(max_quality, node.quality)

        return max_quality

    def update(self, node_expand, reward):
        """
        Backtrack the path from node_expand and update each node until the root
        :param node_expand: the node from wich we launched the simulation
        :param reward: the reward we got
        :return: None
        """
        # mean-update
        current_node = node_expand
        while (current_node.parent != None):
            current_node.update(reward)
            current_node = current_node.parent

    def best_child(self, node):
        """
        Return the best child, based on UCT
        :param node:
        :return: the best child
        """
        best_node = None
        max_score = -float("inf")
        for child in node.generated_children:
            if uct(node, child) > max_score:
                max_score = child.quality
                best_node = child

        return best_node

    def explore_children(self, node, sorted_children):
        """
        Find children of node and add them to sorted_children
        :param node: the parent from which we explore children
        :param sorted_children: PrioritySetQuality.
        :return: None
        """
        for child in node.generated_children:
            sorted_children.add(child)
            self.explore_children(child)


# TODO: command line interface, with pathfile of data, number of patterns and max_time

ITEMS = set()
DATA = read_data('../data/promoters.data')
items = extract_items(DATA)

mcts = MCTSeq(5, ITEMS, DATA, 5, '+')
