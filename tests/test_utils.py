from seqscout.utils import sequence_mutable_to_immutable, \
    sequence_immutable_to_mutable, count_target_class_data, is_subsequence, \
    compute_quality

data = [['+', ({1, 2}, {'num1': 30, 'num2': 40}), ({2}, {'num1': 3, 'num2': 5}), ({1}, {'num1': 10, 'num2': 30.4})],
        ['-', ({1, 2}, {'num1': 30, 'num2': 40}), ({2, 5}, {'num1': 3, 'num2': 5})]]


def test_wracc():
    c = [({1, 2}, {'num1': [30, 31], 'num2': [39, 42]}), ({1}, {'num1': [10, 10], 'num2': [28, 31]})]
    assert compute_quality(data, c, '+') == 0.25


def test_count_target_class_date():
    assert count_target_class_data(data, '+') == 1

def test_is_subsequence():
    a = [({1, 2}, {'num1': 30, 'num2': 40})]
    c = [({1, 2, 3}, {'num1': [30, 31], 'num2': [39, 42]})]
    assert not is_subsequence(c, a)

    a = [({1, 2}, {'num1': 30, 'num2': 40})]
    c = [({1, 2}, {'num1': [30, 31], 'num2': [39, 42]})]
    assert is_subsequence(c, a)

    a = [({1, 2}, {'num1': 30, 'num2': 40}), ({2}, {'num1': 3, 'num2': 5}), ({1}, {'num1': 10, 'num2': 30})]
    c = [({1, 2}, {'num1': [30, 31], 'num2': [39, 42]}), ({1}, {'num1': [10, 10], 'num2': [28, 29]})]
    assert not is_subsequence(c, a)

    a = [({1, 2}, {'num1': 30, 'num2': 40}), ({2}, {'num1': 3, 'num2': 5}), ({1}, {'num1': 10, 'num2': 30.4})]
    c = [({1, 2}, {'num1': [30, 31], 'num2': [39, 42]}), ({1}, {'num1': [10, 10], 'num2': [28, 31]})]
    assert is_subsequence(c, a)

