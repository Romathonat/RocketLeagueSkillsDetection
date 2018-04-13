from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality
from mctseq.utils import extract_items
from mctseq.mctseq_main import MCTSeq

data = [['+', {'A'}, {'B'}]]


def count_mcts_recursive(node, count, known_nodes):
    if node in known_nodes:
        return 0
    else:
        known_nodes.add(node)

    if len(node.generated_children) == 0:
        return 1

    current_count = count
    for child in node.generated_children:
        current_count += count_mcts_recursive(child, count, known_nodes)
    return current_count + 1


def test_permutation_unification():
    # also test if exploration of full latice
    items = extract_items(data)

    mcts = MCTSeq(5, items, data, 1, '+', False)
    mcts.launch()

    # we count elements from the root
    assert count_mcts_recursive(mcts.root_node, 0, set()) == 11


test_permutation_unification()
