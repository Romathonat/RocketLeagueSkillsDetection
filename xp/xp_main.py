from multiprocessing.pool import Pool

import sys

import pandas as pd

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from competitors.beam_search import beam_search
from competitors.misere import misere
from seqehc.misere_hill import misere_hill

from seqehc.utils import read_data, extract_items, \
    encode_items, \
    encode_data, print_results, average_results, read_data_kosarak, \
    read_data_sc2

sys.setrecursionlimit(500000)

# DATA = read_data_kosarak('../data/out.data')
# DATA = read_data_kosarak('../data/aslbu.data')
# DATA = read_data('../data/promoters.data')
# DATA = read_data('../data/splice.data')
# DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]


# third element: enable_i
datasets = [
        (read_data_kosarak('../data/aslbu.data'), '195', False),
        (read_data('../data/promoters.data'), '+', False),
        (read_data('../data/splice.data'), 'EI', False),
        (read_data_kosarak('../data/blocks.data'), '1', False),
        (read_data_kosarak('../data/context.data'), '4', False),
        (read_data_sc2('../data/sequences-TZ-45.txt')[:500], '1', True),
        (read_data_kosarak('../data/skating.data'), '1', True)
]

datasets_names = ['aslbu', 'promoters', 'splice', 'blocks', 'context', 'sc2', 'skating']

def compare_competitors():
    number_dataset = 5
    DATA = datasets[number_dataset][0]
    items = extract_items(DATA)
    target_class = datasets[number_dataset][1]

    TIME = 2

    misere_hill_result = misere_hill(DATA, items, TIME, target_class, top_k=5)
    print_results(misere_hill_result)

    misere_result = misere(DATA, TIME, target_class, 5)
    print_results(misere_result)

    print('beam search')

    beam_results = beam_search(DATA, items, TIME, target_class, False)
    print_results(beam_results)

def compare_datasets():
    pool = Pool(processes=3)
    time_xp = 180

    misere_hist = []
    beam_hist = []
    misere_hill_hist = []

    for i, (data, target, enable_i) in enumerate(datasets):
        items = extract_items(data)

        result_misere = pool.apply_async(misere, (data, time_xp, target))
        result_beam = pool.apply_async(beam_search,
                                       (
                                           data, items, time_xp, target,
                                           enable_i))
        result_misere_hill = pool.apply_async(misere_hill, (
            data, items, time_xp, target, 5, True))

        result_misere = result_misere.get()
        result_beam = result_beam.get()
        result_misere_hill = result_misere_hill.get()

        if len(result_misere) < 5:
            print("Too few example on misere on dataset {}: {} results".format(datasets_names[i], len(result_misere)))

        if len(result_misere_hill) < 5:
            print("Too few example on hillseqs on dataset {}: {} results".format(datasets_names[i], len(result_misere_hill)))

        if len(result_beam) < 5:
            print("Too few example on beam_search on dataset {}: {} results".format(datasets_names[i], len(result_beam)))

        average_misere = average_results(result_misere)
        average_beam = average_results(result_beam)
        average_misere_hill = average_results(result_misere_hill)

        misere_hist.append(average_misere)
        beam_hist.append(average_beam)
        misere_hill_hist.append(average_misere_hill)

    data = {'wracc': misere_hist + misere_hill_hist + beam_hist,
            'dataset': datasets_names + datasets_names + datasets_names,
            'Algorithme': ['misere' for i in range(len(misere_hist))] +
                     ['HillSeqS' for i in range(len(misere_hill_hist))] +
                     ['beam_search' for i in range(len(beam_hist))]}

    df = pd.DataFrame(data=data)

    sns.barplot(x='dataset', y='wracc', hue='Algorithme', data=df)

    plt.show()


def show_quality_over_time():
    # DATA = read_data_kosarak('../data/aslbu.data')
    DATA = read_data('../data/promoters.data')
    # DATA = read_data('../data/splice.data')
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]

    items = extract_items(DATA)
    # items, item_to_encoding, encoding_to_item = encode_items(items)
    # DATA = encode_data(DATA, item_to_encoding)
    target_class = '+'

    pool = Pool(processes=3)
    results_beam = []
    results_beam_50 = []
    results_beam_100 = []

    results_misere = []
    results_misere_hill = []
    x_axis = []

    top_k = 5

    time_step = 17

    xp_time = 10
    for i in range(11):
        print('Iteration: {}'.format(i))
        x_axis.append(xp_time)

        # if we want to average
        nb_launched = 1

        average_nb_launched_beam = 0
        average_nb_launched_beam_50 = 0
        average_nb_launched_beam_100 = 0

        average_nb_launched_misere = 0
        average_nb_launched_misere_hill = 0

        for i in range(nb_launched):
            result_misere = pool.apply_async(misere,
                                             (DATA, xp_time, target_class))
            result_beam = pool.apply_async(beam_search,
                                           (DATA, items, xp_time, target_class,
                                            False, top_k, 30))

            result_beam_50 = pool.apply_async(beam_search,
                                              (DATA, items, xp_time,
                                               target_class,
                                               False, top_k, 50))
            result_beam_100 = pool.apply_async(beam_search,
                                               (DATA, items, xp_time,
                                                target_class,
                                                False, top_k, 100))

            result_misere_hill = pool.apply_async(misere_hill,
                                                  (DATA, items, xp_time,
                                                   target_class, 5))

            average_nb_launched_misere += average_results(result_misere.get())
            average_nb_launched_beam += average_results(result_beam.get())
            average_nb_launched_beam_50 += average_results(
                result_beam_50.get())
            average_nb_launched_beam_100 += average_results(
                result_beam_100.get())
            average_nb_launched_misere_hill += average_results(
                result_misere_hill.get())

        results_beam.append(average_nb_launched_beam / nb_launched)
        results_beam_50.append(average_nb_launched_beam_50 / nb_launched)
        results_beam_100.append(average_nb_launched_beam_100 / nb_launched)
        results_misere.append(average_nb_launched_misere / nb_launched)
        results_misere_hill.append(
            average_nb_launched_misere_hill / nb_launched)

        xp_time += time_step

    '''
    plt.plot(x_axis, results_beam, 'r.-', x_axis, results_misere, 'gx-',
             x_axis, results_misere_hill, 'b+-')

    blue_line = mlines.Line2D([], [], color='blue', marker='+',
                              markersize=5, label='Notre Algo')
    green_line = mlines.Line2D([], [], color='green', marker='x',
                               markersize=5, label='Misere')
    red_line = mlines.Line2D([], [], color='red', marker='.',
                             markersize=5, label='BeamSearch')

    plt.legend(handles=[blue_line, green_line, red_line])

    plt.xlabel('Temps (s)')
    plt.ylabel('Moyenne WRAcc top-5 éléments')
    plt.show()
    '''
    data = {'misere': results_misere,
            'HillSeqS': results_misere_hill, 'beamSearch-30': results_beam,
            'beamSearch-50': results_beam_50,
            'beamSearch-100': results_beam_100}

    df = pd.DataFrame(data=data, index=x_axis)
    ax = sns.lineplot(data=df)

    # ax = sns.lineplot(x='x', y=('misereHill', 'misere'), data=df)

    ax.set(xlabel='Temps(s)', ylabel='Moyenne WRAcc top-5 éléments')

    plt.show()

    with open('./results/result', 'w+') as f:
        f.write(' '.join([str(i) for i in x_axis]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_beam]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere_hill]))


def vertical_vs_horizontal():
    time_xp = 180

    vertical_results = []
    horizontal_results = []


    for data, target, enable_i in datasets:
        items = extract_items(data)

        average_vertical = average_results(
            misere_hill(data, items, time_xp, target, 5, enable_i, False))
        average_horizontal = average_results(
            misere_hill(data, items, time_xp, target, 5, enable_i, True))

        vertical_results.append(average_vertical)
        horizontal_results.append(average_horizontal)

    data = {'wracc': vertical_results + horizontal_results,
            'dataset': datasets_names + datasets_names,
            'Version': ['Vertical' for i in range(len(vertical_results))] +
                     ['Horizontal' for i in range(len(horizontal_results))]}

    df = pd.DataFrame(data=data)

    sns.barplot(x='dataset', y='wracc', hue='Version', data=df)

    plt.show()

def naive_vs_bitset():
    time_xp = 180

    naive_results = []
    bitset_results = []


    for data, target, enable_i in datasets:
        items = extract_items(data)

        average_naive = average_results(
            misere_hill(data, items, time_xp, target, 5, enable_i, wracc_vertical=False))
        average_bitset = average_results(
            misere_hill(data, items, time_xp, target, 5, enable_i, wracc_vertical=True))

        naive_results.append(average_naive)
        bitset_results.append(average_bitset)

    data = {'wracc': naive_results + bitset_results,
            'dataset': datasets_names + datasets_names,
            'Version': ['Naive' for i in range(len(naive_results))] +
                     ['Bitset' for i in range(len(bitset_results))]}

    df = pd.DataFrame(data=data)

    sns.barplot(x='dataset', y='wracc', hue='Version', data=df)

    plt.show()

# vertical_vs_horizontal()
# naive_vs_bitset()
# show_quality_over_time()
compare_competitors()
# compare_datasets()
