#!/usr/bin/env python
# -*- coding: utf-8 -*-


from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, k_length

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]


def test_create_sequence():
    seq = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    assert seq != None
    assert seq.support == 1
    assert seq.quality == 0.25


def test_possible_children():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq3 = SequenceNode([{'A'}, {'B'}], seq1, {'A', 'B'}, data, '+', 0, 2, itemsets_bitsets)

    possible_children = seq3.non_generated_children

    assert sequence_mutable_to_immutable(
        [{'A', 'B'}, {'B'}]) in possible_children
    assert len(possible_children) == 6
    assert seq3.support == 1


def test_possible_children_without_i():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq3 = SequenceNode([{'A'}, {'B'}], seq1, {'A', 'B'}, data, '+', 0, 2, itemsets_bitsets, False)
    possible_children = seq3.non_generated_children
    assert sequence_mutable_to_immutable(
        [{'A'}, {'B'}, {'B'}]) in possible_children

    assert len(possible_children) == 4


def test_update():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq4 = SequenceNode([{'A'}, {'C'}], seq1, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq4.number_visit = 1
    seq4.update(0.5)
    assert seq4.quality == 0.375


def test_expand():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)
    seq4 = SequenceNode([{'A'}, {'C'}], seq1, {'A', 'B', 'C'}, data, '+', 1, 2, itemsets_bitsets)

    non_expanded_children_nb = len(seq4.non_generated_children)
    child = seq4.expand({})

    assert k_length(child.sequence) == k_length(seq4.sequence) + 1
    assert non_expanded_children_nb == len(seq4.non_generated_children) + 1
