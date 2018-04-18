from mctseq.sequencenode import SequenceNode
from mctseq.priorityset import PrioritySetQuality

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]


def test_priorityset():
    root = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    seq4 = SequenceNode([{'B', 'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})

    child = SequenceNode([{'A'}, {'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    child2 = SequenceNode([{'A'}, {'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    child3 = SequenceNode([{'A'}, {'B', 'C'}], None, {'A', 'B', 'C'}, data,
                          '+', 1, 2, {})

    priority = PrioritySetQuality()
    priority.add(root)
    priority.add(child)
    priority.add(child2)
    priority.add(child3)

    # _, best_element = priority.get()
    _, best_element = priority.get_top_k(1)[0]

    assert best_element == child


def test_unique():
    root = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    child = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})

    priority = PrioritySetQuality()
    priority.add(root)
    priority.add(child)

    assert len(priority.heap) == 1
