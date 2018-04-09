from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality
from mctseq.utils import extract_items
from mctseq.mctseq_main import MCTSeq

data = [['+', {'A'}, {'B'}]]


def test_permutation_unification():
    # also test if exploration of full latice
    items = extract_items(data)

    mcts = MCTSeq(5, items, data, 1, '+', False)
    mcts.launch()

    # we count elements from the root
    assert len(mcts.node_hashmap) == 11

