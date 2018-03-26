# -*- coding: utf-8 -*-

"""Main module."""
import datetime

from mctseq.utils import read_data, extract_items, uct
from mctseq.SequenceNode import SequenceNode


class MCTSeq():
    def __init__(self, pattern_number, items, time_budget):
        self.pattern_number = pattern_number
        self.items = items
        self.time_budget = time_budget

    def launch(self, pattern_number):
        """
        Launch the algorithm, specifying how many patterns we want to mine
        :return:
        """
        begin = datetime.datetime.utcnow()

        # current_node is root
        root_node = SequenceNode([], None, items)
        current_node = root_node

        while datetime.datetime.utcnow() - begin < self.time_budget:
            node_sel = self.select(current_node)
            node_expand = self.expand(node_sel)
            reward = self.roll_out(node_expand)
            self.update(node_expand, reward)

    def select(self, node):
        """
        Select the best node, using exploration-exploitation tradeoff
        :param node: the node from where we begin to search
        :return: the selected node
        """

    def expand(self, node):
        """
        Choose a child to expand among possible children of node
        :param node: the node from wich we want to expand
        :return: the expanded node
        """
        pass

    def roll_out(self, node):
        """
        Equivalent to simulation in classical MCTS
        :param node: the node from wich launch the roll_out
        :return: the quality measure we desire
        """
        pass

    def update(self, node_expand, reward):
        """
        Backtrack the path from node_expand and update each node until the root
        :param node_expand: the node from wich we
        :param reward: None
        :return:
        """
        pass

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


# Todo: command line interface, with pathfile of data

ITEMS = set()
DATA = read_data('/home/romain/Documents/contextPrefixSpan.txt')
items = extract_items(data)
