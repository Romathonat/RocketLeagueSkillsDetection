import pathlib

from seqscout.seq_scout import seq_scout_api
from seqscout.utils import read_data

if __name__ == '__main__':
    DATA = read_data(pathlib.Path(__file__).parent / 'data/promoters.data')
    seq_scout_api(DATA, '+', 10, 5)
