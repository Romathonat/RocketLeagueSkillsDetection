from seqscout.utils import sequence_mutable_to_immutable, \
    sequence_immutable_to_mutable, count_target_class_data, is_subsequence, \
    following_ones, generate_bitset, create_s_extension, create_i_extension, \
    get_support_from_vector, compute_bitset_slot_size, compute_quality, \
    compute_quality_vertical, jaccard_measure, find_LCS

data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}]]
first_zero_mask = int('0101', 2)
last_ones_mask = int('0101', 2)
bitset_slot_size = 2

kwargs = {'first_zero_mask': first_zero_mask, 'last_ones_mask': last_ones_mask,
          'bitset_slot_size': bitset_slot_size, 'node_hashmap': {}}


def test_wracc():
    assert compute_quality(data, [{'B'}], '-') == 0
    assert compute_quality(data, [{'C'}], '+') == 0.25


def test_wracc_vertical():
    assert compute_quality_vertical(data, [{'B'}], '-', bitset_slot_size, {}, 1,
                                    first_zero_mask, last_ones_mask)[0] == 0
    assert compute_quality_vertical(data, [{'C'}], '+', bitset_slot_size, {}, 1,
                                    first_zero_mask, last_ones_mask)[0] == 0.25


def test_jaccard_bitset():
    first_zero_mask = int('010101', 2)
    last_ones_mask = int('010101', 2)
    bitset_slot_size = 2
    data = [['+', {'A', 'B'}, {'C'}], ['-', {'A'}, {'B'}], ['+', {'C'}]]

    assert jaccard_measure(int('010010', 2), int('100100', 2), bitset_slot_size, first_zero_mask, last_ones_mask) == 1/3


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


def test_compute_bitset_slot_size():
    assert compute_bitset_slot_size(data) == 2


def test_is_subsequence():
    a = ({1, 2}, {2, 3})
    b = ({1, 2, 3}, {2, 4, 3})
    c = ({1}, {2}, {2})

    assert not is_subsequence(c, a)

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

    a = sequence_mutable_to_immutable(a)
    b = sequence_mutable_to_immutable(b)
    assert is_subsequence(a, b)

    a = [{'1'}, {'2'}]
    b = [{'1', '2', '3'}, {'1'}, {'2', '4', '3'}]

    assert is_subsequence(a, b)


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

    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (
    2, int('11', 2))

    a = int('00100000', 2)
    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (
    1, int('10', 2))

    a = int('00010001', 2)
    assert get_support_from_vector(a, 4, zero_mask, ones_mask) == (
    2, int('11', 2))


def test_generate_bitset():
    bitset = generate_bitset({'A'}, data, 2)
    assert bitset == int('1010', 2)

    bitset = generate_bitset({'A'}, data, 4)
    assert bitset == int('10001000', 2)


def test_lcs():
    seq1 = [{'a', 'b'}, {'e'}, {'c'}]
    seq2 = [{'a' }, {'d'}, {'a', 'b'}, {'f'}, {'e'}]

    lcs = find_LCS(seq1, seq2)

    assert lcs == [{'a', 'b'}, {'e'}]

