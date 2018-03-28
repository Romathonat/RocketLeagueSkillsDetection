from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, \
    sequence_immutable_to_mutable, count_target_class_data, is_subsequence

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]


def test_sequence_mutable_to_imutable():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'}, data, '+', 0)
    immutable = sequence_mutable_to_immutable(seq.sequence)

    assert len(immutable) == 2
    assert isinstance(immutable, tuple)


def test_sequence_imutable_to_mutable():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'}, data, '+', 0)
    immutable = sequence_mutable_to_immutable(seq.sequence)

    mutable = sequence_immutable_to_mutable(immutable)

    assert len(mutable) == 2
    assert isinstance(mutable, list)


def test_count_target_class_date():
    assert count_target_class_data(data, '+') == 1


def test_is_subsequence():
    a = ({1, 2}, {2, 3})
    b = ({1, 2, 3}, {2, 4, 3})

    assert is_subsequence(a, b)

    a = [{1, 2}, {2, 3}]
    b = [{1, 2, 3}, {1}, {2, 4, 3}]

    assert is_subsequence(a, b)

    a = [{1, 5, 2}, {2, 3}]
    b = [{1, 2, 3}, {1}, {2, 4, 3}]

    assert not is_subsequence(a, b)

    a = [{1, 5, 2}, {2, 3}, {5}]
    b = [{1, 5, 2}, {2, 4, 3}]

    assert not is_subsequence(a, b)

    a = [{1}, {2}]
    b = [{1, 2, 3}, {1}, {2, 4, 3}]

    assert is_subsequence(a, b)
    assert not is_subsequence(b, a)
