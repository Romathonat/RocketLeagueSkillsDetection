from seqscout.priorityset import PrioritySet, PrioritySetUCB
from seqscout.seq_scout import UCB

def test_over_top_k():
    priority = PrioritySet()
    priority.add(frozenset([1, 2]), 0.1)

    assert len(priority.get_top_k(2)) == 1

def test_UCB():
    priority = PrioritySetUCB()
    UCB_1 = UCB(0.1, 5, 6)
    UCB_2 = UCB(0.05, 1, 6)

    priority.add(frozenset([1, 2]), (UCB_1, 5, 0.1))
    priority.add(frozenset([1, 3]), (UCB_2, 1, 0.05))
    priority.add(frozenset([1, 3]), (1.5, 1, 0.05))

    assert round(UCB_1, 3) == 1.123
    assert UCB_2 > UCB_1

    assert round(priority.pop()[0], 2) == 1.55
