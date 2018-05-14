from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, \
    sequence_immutable_to_mutable, count_target_class_data, is_subsequence, \
    following_ones, generate_bitset, create_s_extension, create_i_extension, \
    get_support_from_vector

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]
first_zero_mask = int('0101', 2)
first_zero_mask = int('0101', 2)
last_ones_mask = int('0101', 2)
bitset_slot_size = 2

kwargs = {'first_zero_mask': first_zero_mask, 'last_ones_mask': last_ones_mask,
          'bitset_slot_size': bitset_slot_size}


def test_sequence_mutable_to_imutable():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'}, data, '+', 0,
                       {}, **kwargs)
    immutable = sequence_mutable_to_immutable(seq.sequence)

    assert len(immutable) == 2
    assert isinstance(immutable, tuple)


def test_sequence_imutable_to_mutable():
    seq = SequenceNode([{'A'}, {'BC'}], None, {'A', 'B', 'C'}, data, '+', 0,
                       {}, **kwargs)
    immutable = sequence_mutable_to_immutable(seq.sequence)

    mutable = sequence_immutable_to_mutable(immutable)

    assert len(mutable) == 2
    assert isinstance(mutable, list)


def test_create_s_extension():
    sequence = (frozenset({'A'}), frozenset({'C'}))
    s_ext = create_s_extension(sequence, 'B', 0)
    assert s_ext == (frozenset({'B'}), frozenset({'A'}), frozenset({'C'}))

    s_ext = create_s_extension(sequence, 'B', 1)
    assert s_ext == (frozenset({'A'}), frozenset({'B'}), frozenset({'C'}))

    s_ext = create_s_extension(sequence, 'B', 2)
    assert s_ext == (frozenset({'A'}), frozenset({'C'}), frozenset({'B'}))


def test_create_i_extension():
    sequence = (frozenset({'A'}), frozenset({'C'}))

    s_ext = create_i_extension(sequence, 'B', 0)
    assert s_ext == (frozenset({'A', 'B'}), frozenset({'C'}))

    s_ext = create_i_extension(sequence, 'B', 1)
    assert s_ext == (frozenset({'A'}), frozenset({'B', 'C'}))


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


def test_following_ones():
    a = int('01000010', 2)
    zero_mask = int('01110111', 2)
    assert following_ones(a, 4, zero_mask) == int('00110001', 2)

    a = int('00011111', 2)
    assert following_ones(a, 4, zero_mask) == int('00000111', 2)

    a = int('10000000')
    assert following_ones(a, 4, zero_mask) == int('01110000', 2)

    a = int('00000001')
    assert following_ones(a, 4, zero_mask) == int('00000000', 2)


def test_get_support_from_vector():
    a = int('01000010', 2)
    zero_mask = int('01110111', 2)
    ones_mask = int('00010001', 2)

    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (2, int('11', 2))

    a = int('00100000', 2)
    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (1, int('10', 2))

    a = int('00010001', 2)
    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (2, int('11', 2))


def test_generate_bitset():
    bitset = generate_bitset({'A'}, data, 2)
    assert bitset == int('1010', 2)

    bitset = generate_bitset({'A'}, data, 4)
    assert bitset == int('10001000', 2)
