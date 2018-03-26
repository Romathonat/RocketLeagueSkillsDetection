#!/usr/bin/env python
# -*- coding: utf-8 -*-


from mctseq.SequenceNode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable


def test_create_sequence():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'})
    assert seq != None


def test_sequence_mutable_to_imutable():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'})
    immutable = sequence_mutable_to_immutable(seq.sequence)

    assert len(immutable) == 2
    assert isinstance(immutable, tuple)


def test_possible_children():
    seq = SequenceNode([{'A'}, {'B'}], None, {'A', 'B'})
    possible_children = seq.possible_children
    assert sequence_mutable_to_immutable(
        [{'A', 'B'}, {'B'}]) in possible_children
    assert len(possible_children) == 6
    # check if no duplicate
