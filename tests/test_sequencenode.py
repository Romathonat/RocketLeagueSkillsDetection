#!/usr/bin/env python
# -*- coding: utf-8 -*-


from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, k_length

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]


def test_create_sequence():
    seq = SequenceNode([{'A'}, {'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    assert seq != None
    assert seq.support == 1
    assert seq.quality == 0.25


def test_possible_children():
    seq = SequenceNode([{'A'}, {'B'}], None, {'A', 'B'}, data, '+', 0, 2, {})
    possible_children = seq.non_generated_children
    assert sequence_mutable_to_immutable(
        [{'A', 'B'}, {'B'}]) in possible_children
    assert len(possible_children) == 6


def test_possible_children_without_i():
    seq = SequenceNode([{'A'}, {'B'}], None, {'A', 'B'}, data, '+', 0, 2, {}, False)
    possible_children = seq.non_generated_children
    assert sequence_mutable_to_immutable(
        [{'A'}, {'B'}, {'B'}]) in possible_children

    assert len(possible_children) == 4


def test_update():
    seq = SequenceNode([{'A'}, {'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    seq.number_visit = 1
    seq.update(0.5)
    assert seq.quality == 0.375


def test_expand():
    seq = SequenceNode([{'A'}, {'C'}], None, {'A', 'B', 'C'}, data, '+', 1, 2, {})
    non_expanded_children_nb = len(seq.non_generated_children)
    child = seq.expand({})

    assert k_length(child.sequence) == k_length(seq.sequence) + 1
    assert non_expanded_children_nb == len(seq.non_generated_children) + 1
