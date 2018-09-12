from multiprocessing.pool import Pool

import sys

import matplotlib.pyplot as plt

from competitors.misere import misere
from seqehc.misere_hill import misere_hill

from seqehc.utils import read_data, extract_items, \
    encode_items, \
    encode_data, print_results, average_results

sys.setrecursionlimit(500000)

# DATA = read_data_kosarak('../data/out.data')
DATA = read_data('../data/promoters.data')
# DATA = read_data('../data/splice.data')
# DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:300]

items = extract_items(DATA)
items, item_to_encoding, encoding_to_item = encode_items(items)
DATA = encode_data(DATA, item_to_encoding)
target_class = '+'

def misere_vs_misere_hill():
    TIME = 10

    misere_hill_result = misere_hill(DATA,items ,TIME, target_class, 5)
    print_results(misere_hill_result)

    misere_result = misere(DATA, TIME, target_class, 5)
    print_results(misere_result)


def show_quality_over_time():
    results_mcts = []
    results_misere = []
    x_axis = []

    pool = Pool(processes=2)
    time_step = 5

    # 10 data points
    xp_time = 3
    for i in range(1, 11):
        print('Iteration: {}'.format(i))
        x_axis.append(xp_time)

        # we average on 3 distinct launch
        nb_launched = 3
        average_nb_launched_mcts = 0
        average_nb_launched_misere = 0

        for i in range(nb_launched):
            mcts = MCTSeq(10, items, DATA, xp_time, target_class, enable_i=False)
            result_mcts = pool.apply_async(mcts.launch)
            result_misere = pool.apply_async(misere,
                                              (DATA, xp_time, target_class))

            average_nb_launched_mcts += average_results(result_mcts.get())
            average_nb_launched_misere += average_results(result_misere.get())

        results_mcts.append(average_nb_launched_mcts / nb_launched)
        results_misere.append(average_nb_launched_misere / nb_launched)

        xp_time +=  time_step


    plt.plot(x_axis, results_mcts, 'ro-', x_axis, results_misere, 'go-')
    plt.show()

    with open('../xp/results/result', 'w+') as f:
        f.write(x_axis)
        f.write(results_mcts)
        f.write(results_misere)

#show_quality_over_time()
#basic_xp_2()
misere_vs_misere_hill()
