from datetime import datetime
from multiprocessing.pool import Pool

from mctseq.mctseq_main import MCTSeq
from competitors.misere import misere
from mctseq.utils import read_data, read_data_kosarak, extract_items, encode_items, \
    encode_data, print_results_mcts, print_results

import sys

sys.setrecursionlimit(500000)

#DATA = read_data_kosarak('../data/out.data')
DATA = read_data('../data/promoters.data')
#DATA = read_data('../data/splice.data')

items = extract_items(DATA)
items, item_to_encoding, encoding_to_item = encode_items(items)
DATA = encode_data(DATA, item_to_encoding)
TIME = 50
target_class = '+'


pool = Pool(processes=2)

mcts = MCTSeq(10, items, DATA, TIME, target_class, enable_i=False)

result_mcts = pool.apply_async(mcts.launch)

results_misere = pool.apply_async(misere, (DATA, TIME, target_class))

print('MCTS')
print_results_mcts(result_mcts.get(), encoding_to_item)
print('Misere')
print_results(results_misere.get())
