import datetime
import random
import copy
import pathlib

import math
import os

from seqsamphill.utils import read_data, read_data_kosarak, uct, \
    is_subsequence, sequence_mutable_to_immutable, print_results, \
    read_data_sc2, k_length, generate_bitset, following_ones, \
    get_support_from_vector, compute_first_zero_mask, compute_last_ones_mask, \
    count_target_class_data, extract_items, compute_WRAcc, compute_WRAcc_vertical, jaccard_measure, find_LCS, \
    reduce_k_length, average_results, sequence_immutable_to_mutable

from seqsamphill.priorityset import PrioritySet, PrioritySetUCB

from mctsextend.node import Node


def best_child(node):
    best_node = None
    max_score = -float("inf")

    for child in node.children:
        current_ucb = child.get_normalized_wracc() / child.number_visits + 0.5 * math.sqrt(
            2 * math.log(node.number_visits) / child.number_visits)

        if current_ucb > max_score:
            max_score = current_ucb
            best_node = child

    return best_node


def select(node, data):
    """
    Select the best node, using exploration-exploitation tradeoff
    :param node: the node from where we begin to search
    :return: the selected node, or None if exploration is finished
    """
    # we reach a terminal state if the extend corresponds to the whole database
    while len(node.extend) < len(data):
        # the node is fully expanded if its variable i_expand reached the end
        if node.i_expand < len(data) - 1:
            return node
        else:
            node = best_child(node)

    # if we reach this point, it means we reached a terminal node: a node without children
    return node


def expand(node, data, target_class):
    for i in range(node.i_expand, len(data)):
        object = data[i]
        # removing class
        object = object[1:]
        object = sequence_mutable_to_immutable(object)

        if i not in node.extend:
            try:
                expanded_child = Node(object, node, data, target_class)
                node.i_expand = i + 1
                return expanded_child
            except ValueError:
               continue

    node.i_expand = len(data)

    return "fully_expanded"


def roll_out(node, data, target_class):
    sequence = copy.deepcopy(node.intend)
    sequence = sequence_immutable_to_mutable(sequence)

    # we remove z items randomly
    seq_items_nb = len([i for j_set in sequence for i in j_set])
    z = random.randint(0, seq_items_nb - 1)

    for _ in range(z):
        chosen_itemset_i = random.randint(0, len(sequence) - 1)
        chosen_itemset = sequence[chosen_itemset_i]

        chosen_itemset.remove(random.sample(chosen_itemset, 1)[0])

        if len(chosen_itemset) == 0:
            sequence.pop(chosen_itemset_i)

    reward = compute_WRAcc(data, sequence, target_class)

    return sequence, reward


def update(node, reward):
    """
    Backtrack: update the node and recursively update all nodes until the root
    :param node: the node we want to update
    :param reward: the reward we got
    :return: None
    """
    node.update(reward)

    for parent in node.parents:
        if parent != None:
            update(parent, reward)

def select_expand(root_node, data, target_class, sorted_patterns):
    node_sel = select(root_node, data)
    if len(node_sel.extend) == data:
        # we choosed a terminal node, we update ucb with a terrible score and relaunch the select
        update(node_sel, -float('inf'))
        return select_expand(root_node, data, target_class, sorted_patterns)

    node_expand = expand(node_sel, data, target_class)
    if isinstance(node_expand, str):
        # node_sel is fully expanded already, relaunch and add a terrible score
        update(node_sel, -float('inf'))
        return select_expand(root_node, data, target_class, sorted_patterns)
    else:
        sorted_patterns.add(node_expand.intend, node_expand.quality)

    return node_expand

def launch_mcts(data, time_budget, target_class, top_k=10, iterations_limit=float('inf')):
    begin = datetime.datetime.utcnow()
    root_node = Node(None, None, data, target_class)
    # node_hashmap = {}
    root_pattern = sequence_mutable_to_immutable(root_node.intend)
    # node_hashmap[root_pattern] = root_node

    sorted_patterns = PrioritySet()

    iteration_count = 0

    end_time = begin + datetime.timedelta(seconds=time_budget)

    while datetime.datetime.utcnow() <= end_time and iteration_count < iterations_limit:
        node_expand = select_expand(root_node, data, target_class, sorted_patterns)

        sequence_reward, reward = roll_out(node_expand, data, target_class)
        sorted_patterns.add(sequence_mutable_to_immutable(sequence_reward), reward)

        update(node_expand, reward)

        iteration_count += 1

    print('Number iteration mcts: {}'.format(iteration_count))

    return sorted_patterns.get_top_k_non_redundant(data, top_k)

# TODO: memory preservation
# TODO: case of pattern with no children
if __name__ == '__main__':
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]
    # DATA = read_data(pathlib.Path(__file__).parent.parent / 'data/promoters.data')
    #DATA = [i[:5] for i in DATA]

    results = launch_mcts(DATA, 12, '1', top_k=10, iterations_limit=200)
    print_results(results)
