
from mctseq.sequencenode import SequenceNode
from mctseq.utils import sequence_mutable_to_immutable, sequence_immutable_to_mutable

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

