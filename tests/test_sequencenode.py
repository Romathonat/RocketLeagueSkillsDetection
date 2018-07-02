#!/usr/bin/env python
# -*- coding: utf-8 -*-


from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, k_length

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]
first_zero_mask = int('0101', 2)
last_ones_mask = int('0101', 2)
bitset_slot_size = 2
node_hashmap = {}

kwargs = {'first_zero_mask': first_zero_mask, 'last_ones_mask': last_ones_mask,
          'bitset_slot_size': bitset_slot_size, 'node_hashmap': node_hashmap}


def test_create_sequence():
    seq = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1, {},
                       **kwargs)
    assert seq != None
    assert seq.support == 1
    assert seq.wracc == 0.25
    assert seq.quality == 1


def test_possible_children():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq3 = SequenceNode([{'A'}, {'B'}], seq1, {'A', 'B'}, data, '+', 0,
                        itemsets_bitsets, **kwargs)

    possible_children = seq3.non_generated_children

    assert sequence_mutable_to_immutable(
        [{'A', 'B'}, {'B'}]) in possible_children
    assert len(possible_children) == 6
    assert seq3.support == 1


def test_possible_children_without_i():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq3 = SequenceNode([{'A'}, {'B'}], seq1, {'A', 'B'}, data, '+', 0,
                        itemsets_bitsets, enable_i=False, **kwargs)
    possible_children = seq3.non_generated_children
    assert sequence_mutable_to_immutable(
        [{'A'}, {'B'}, {'B'}]) in possible_children

    assert len(possible_children) == 4


def test_update():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq4 = SequenceNode([{'A'}, {'C'}], seq1, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq4.number_visit = 1
    seq4.update(0.5)
    assert seq4.quality == 0.75


def test_expand():
    itemsets_bitsets = {}
    seq1 = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq2 = SequenceNode([{'B'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq3 = SequenceNode([{'C'}], None, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)
    seq4 = SequenceNode([{'A'}, {'C'}], seq1, {'A', 'B', 'C'}, data, '+', 1,
                        itemsets_bitsets, **kwargs)

    non_expanded_children_nb = len(seq4.non_generated_children)
    child = seq4.expand({})

    assert k_length(child.sequence) == k_length(seq4.sequence) + 1
    assert non_expanded_children_nb == len(seq4.non_generated_children) + 1


def test_quality():
    data_local = [['+', {'A', 'B'}, {'A', 'C'}], ['+', {'A'}], ['+', {'B'}],
                  ['+', {'A', 'B'}], ['-', {'A'}, {'B'}]]

    first_zero_mask = int('0101010101', 2)
    last_ones_mask = int('0101010101', 2)
    bitset_slot_size = 2

    kwargs = {'first_zero_mask': first_zero_mask,
              'last_ones_mask': last_ones_mask,
              'bitset_slot_size': bitset_slot_size,
              'node_hashmap': node_hashmap}

    itemsets_bitsets = {}
    seq = SequenceNode([{'A'}], None, {'A', 'B', 'C'}, data_local, '+', 4,
                       itemsets_bitsets, **kwargs)
    seq.number_visit = 1
    assert seq.class_pattern_count == 3
