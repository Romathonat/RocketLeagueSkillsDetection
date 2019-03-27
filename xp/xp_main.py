from multiprocessing.pool import Pool

import sys

import pandas as pd

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from competitors.beam_search import beam_search
from competitors.misere import misere
from competitors.seed_explore import seed_explore
from competitors.seed_explore_v2 import seed_explore_v2
from seqsamphill.seq_samp_hill import seq_samp_hill

from seqsamphill.utils import read_data, extract_items, \
    encode_items, \
    encode_data, print_results, average_results, read_data_kosarak, \
    read_data_sc2, reduce_k_length

from competitors.seed_explore import seed_explore

sys.setrecursionlimit(500000)

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


def compare_seeds(number_dataset):
    DATA = datasets[number_dataset][0]
    items = extract_items(DATA)
    target_class = datasets[number_dataset][1]
    enable_i = datasets[number_dataset][2]

    TIME = 20

    print('Dataset: {}'.format(datasets_names[number_dataset]))
    seq_samp_hill_results = seq_samp_hill(DATA, items, TIME, target_class, top_k=5, enable_i=enable_i)
    print_results(seq_samp_hill_results)

    seed_results = seed_explore(DATA, items, TIME, target_class, top_k=5, enable_i=enable_i)
    print_results(seed_results)


#compare_seeds(0)
#compare_seeds(1)
#compare_seeds(2)
#compare_seeds(3)
#compare_seeds(4)
#compare_seeds(5)
#compare_seeds(6)

def compare_competitors():
    number_dataset = 5
    DATA = datasets[number_dataset][0]
    items = extract_items(DATA)
    target_class = datasets[number_dataset][1]
    enable_i = datasets[number_dataset][2]

    TIME = 180

    seq_samp_hill_results = seq_samp_hill(DATA, items, TIME, target_class, top_k=5, enable_i=enable_i)
    print_results(seq_samp_hill_results)

    misere_result = misere(DATA, TIME, target_class, 5)
    print_results(misere_result)

    print('beam search')
    beam_results = beam_search(DATA, items, TIME, target_class, enable_i=enable_i)
    print_results(beam_results)


def compare_datasets():
    pool = Pool(processes=3)
    time_xp = 10
    top_k = 15

    misere_hist = []
    beam_hist = []
    seq_samp_hill_hist = []

    for i, (data, target, enable_i) in enumerate(datasets):
        items = extract_items(data)

        result_misere = pool.apply_async(misere, (data, time_xp, target, top_k))
        result_beam = pool.apply_async(beam_search,
                                       (
                                           data, items, time_xp, target,
                                           enable_i, top_k))
        result_seq_samp_hill = pool.apply_async(seq_samp_hill, (
            data, items, time_xp, target, top_k, True))

        result_misere = result_misere.get()
        result_beam = result_beam.get()
        result_seq_samp_hill = result_seq_samp_hill.get()

        if len(result_misere) < 5:
            print("Too few example on misere on dataset {}: {} results".format(datasets_names[i], len(result_misere)))

        if len(result_seq_samp_hill) < 5:
            print("Too few example on SeqSampHill on dataset {}: {} results".format(datasets_names[i],
                                                                                    len(result_seq_samp_hill)))

        if len(result_beam) < 5:
            print(
                "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i], len(result_beam)))

        average_misere = average_results(result_misere)
        average_beam = average_results(result_beam)
        average_seq_samp_hill = average_results(result_seq_samp_hill)

        misere_hist.append(average_misere)
        beam_hist.append(average_beam)
        seq_samp_hill_hist.append(average_seq_samp_hill)

    data = {'WRAcc': misere_hist + seq_samp_hill_hist + beam_hist,
            'dataset': datasets_names + datasets_names + datasets_names,
            'Algorithm': ['misere' for i in range(len(misere_hist))] +
                         ['SeqSampHill' for i in range(len(seq_samp_hill_hist))] +
                         ['beam_search' for i in range(len(beam_hist))]}

    df = pd.DataFrame(data=data)

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/barplot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        f.write(' '.join([str(i) for i in seq_samp_hill_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    # plt.show()


def compare_datasets_seeds():
    pool = Pool(processes=3)
    time_xp = 120
    top_k = 10

    misere_hist = []
    beam_hist = []
    seq_samp_hill_hist = []
    seeds_hist = []

    for i, (data, target, enable_i) in enumerate(datasets):
        items = extract_items(data)

        result_misere = pool.apply_async(misere, (data, time_xp, target, top_k))
        result_beam = pool.apply_async(beam_search,
                                       (
                                           data, items, time_xp, target,
                                           enable_i, top_k))

        result_seq_samp_hill = pool.apply_async(seq_samp_hill, (
            data, items, time_xp, target, top_k, enable_i))

        result_seeds = pool.apply_async(seed_explore, (data, items, time_xp, target, top_k, enable_i))

        result_misere = result_misere.get()
        result_beam = result_beam.get()
        result_seq_samp_hill = result_seq_samp_hill.get()
        result_seeds = result_seeds.get()

        if len(result_misere) < top_k:
            print("Too few example on misere on dataset {}: {} results".format(datasets_names[i], len(result_misere)))

        if len(result_seq_samp_hill) < top_k:
            print("Too few example on SeqSampHill on dataset {}: {} results".format(datasets_names[i],
                                                                                    len(result_seq_samp_hill)))

        if len(result_beam) < top_k:
            print(
                "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i], len(result_beam)))
        if len(result_seeds) < top_k:
            print(
                "Too few example on seeds explore on dataset {}: {} results".format(datasets_names[i],
                                                                                    len(result_seeds)))

        average_misere = average_results(result_misere)
        average_beam = average_results(result_beam)
        average_seq_samp_hill = average_results(result_seq_samp_hill)
        average_seeds = average_results(result_seeds)

        misere_hist.append(average_misere)
        beam_hist.append(average_beam)
        seq_samp_hill_hist.append(average_seq_samp_hill)
        seeds_hist.append(average_seeds)

    data = {'WRAcc': misere_hist + seq_samp_hill_hist + beam_hist + seeds_hist,
            'dataset': datasets_names + datasets_names + datasets_names + datasets_names,
            'Algorithm': ['misere' for i in range(len(misere_hist))] +
                         ['SeqSampHill' for i in range(len(seq_samp_hill_hist))] +
                         ['beam_search' for i in range(len(beam_hist))] +
                         ['Seeds Explore' for i in range(len(seeds_hist))]}

    df = pd.DataFrame(data=data)

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/barplot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        f.write(' '.join([str(i) for i in seq_samp_hill_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    # plt.show()

def show_quality_over_time():
    # DATA = read_data_kosarak('../data/aslbu.data')
    DATA = read_data('../data/promoters.data')
    # DATA = read_data('../data/splice.data')
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:500]

    items = extract_items(DATA)
    # items, item_to_encoding, encoding_to_item = encode_items(items)
    # DATA = encode_data(DATA, item_to_encoding)
    target_class = '1'

    pool = Pool(processes=3)
    results_beam = []
    results_beam_50 = []
    results_beam_100 = []

    results_misere = []
    results_seq_samp_hill = []
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
                                             (DATA, xp_time, target_class, top_k))
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

            result_misere_hill = pool.apply_async(seq_samp_hill,
                                                  (DATA, items, xp_time,
                                                   target_class, top_k))

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
        results_seq_samp_hill.append(
            average_nb_launched_misere_hill / nb_launched)

        xp_time += time_step

    data = {'misere': results_misere,
            'SeqSampHill': results_seq_samp_hill, 'beamSearch-30': results_beam,
            'beamSearch-50': results_beam_50,
            'beamSearch-100': results_beam_100}

    df = pd.DataFrame(data=data, index=x_axis)

    plt.clf()
    ax = sns.lineplot(data=df)

    # ax = sns.lineplot(x='x', y=('misereHill', 'misere'), data=df)

    ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-5 patterns')

    plt.savefig('./time/promoters_over_time.png')

    with open('./time/result', 'w+') as f:
        f.write(' '.join([str(i) for i in x_axis]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_beam]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_seq_samp_hill]))

    # plt.show()


def compare_ground_truth():
    DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    DATA = reduce_k_length(10, DATA)

    items = extract_items(DATA)

    target_class = '1'

    pool = Pool(processes=3)

    results_beam = []
    results_beam_50 = []
    results_beam_100 = []

    results_misere = []
    results_seq_samp_hill = []
    x_axis = []

    top_k = 10
    time_step = 17

    xp_time = 10

    # found with exaustive search
    ground_truth = 0.008893952000000009

    for i in range(11):
        print('Iteration: {}'.format(i))
        x_axis.append(xp_time)

        # if we want to average
        nb_launched = 5

        average_nb_launched_beam = 0
        average_nb_launched_beam_50 = 0
        average_nb_launched_beam_100 = 0

        average_nb_launched_misere = 0
        average_nb_launched_seq_samp_hill = 0

        for i in range(nb_launched):
            result_misere = pool.apply_async(misere,
                                             (DATA, xp_time, target_class, top_k))
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

            result_seq_samp_hill = pool.apply_async(seq_samp_hill,
                                                    (DATA, items, xp_time,
                                                     target_class, top_k))

            average_nb_launched_misere += average_results(result_misere.get())
            average_nb_launched_beam += average_results(result_beam.get())
            average_nb_launched_beam_50 += average_results(
                result_beam_50.get())
            average_nb_launched_beam_100 += average_results(
                result_beam_100.get())
            average_nb_launched_seq_samp_hill += average_results(
                result_seq_samp_hill.get())

        results_beam.append(average_nb_launched_beam / ground_truth / nb_launched)
        results_beam_50.append(average_nb_launched_beam_50 / ground_truth / nb_launched)
        results_beam_100.append(average_nb_launched_beam_100 / ground_truth / nb_launched)
        results_misere.append(average_nb_launched_misere / ground_truth / nb_launched)

        results_seq_samp_hill.append(
            average_nb_launched_seq_samp_hill / nb_launched / ground_truth)

        xp_time += time_step

    data = {'misere': results_misere,
            'SeqSampHill': results_seq_samp_hill, 'beamSearch-30': results_beam,
            'beamSearch-50': results_beam_50,
            'beamSearch-100': results_beam_100}

    df = pd.DataFrame(data=data, index=x_axis)

    plt.clf()
    ax = sns.lineplot(data=df)

    # ax = sns.lineplot(x='x', y=('misereHill', 'misere'), data=df)

    ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./ground_truth/gt.png')

    with open('./ground_truth/result', 'w+') as f:
        f.write(' '.join([str(i) for i in x_axis]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_beam]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_seq_samp_hill]))

    # plt.show()


def quality_over_dataset_size():
    DATA_origin = read_data('../data/promoters.data')
    # DATA_origin = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]

    # DATA = reduce_k_length(10, DATA)

    items = extract_items(DATA_origin)

    target_class = '+'

    pool = Pool(processes=3)

    results_beam = []
    results_beam_50 = []
    results_beam_100 = []

    results_misere = []
    results_seq_samp_hill = []
    x_axis = []

    top_k = 10
    size_step = 4
    size = 5

    xp_time = 10

    for i in range(1, 15):
        print('Iteration: {}'.format(i))
        x_axis.append(size)

        DATA = reduce_k_length(size, DATA_origin)

        # if we want to average
        nb_launched = 5

        average_nb_launched_beam = 0
        average_nb_launched_beam_50 = 0
        average_nb_launched_beam_100 = 0

        average_nb_launched_misere = 0
        average_nb_launched_seq_samp_hill = 0

        for i in range(nb_launched):
            result_misere = pool.apply_async(misere,
                                             (DATA, xp_time, target_class, top_k))
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

            result_seq_samp_hill = pool.apply_async(seq_samp_hill,
                                                    (DATA, items, xp_time,
                                                     target_class, top_k))

            average_nb_launched_misere += average_results(result_misere.get())
            average_nb_launched_beam += average_results(result_beam.get())
            average_nb_launched_beam_50 += average_results(
                result_beam_50.get())
            average_nb_launched_beam_100 += average_results(
                result_beam_100.get())
            average_nb_launched_seq_samp_hill += average_results(
                result_seq_samp_hill.get())

        results_beam.append(average_nb_launched_beam / nb_launched)
        results_beam_50.append(average_nb_launched_beam_50 / nb_launched)
        results_beam_100.append(average_nb_launched_beam_100 / nb_launched)
        results_misere.append(average_nb_launched_misere / nb_launched)

        results_seq_samp_hill.append(
            average_nb_launched_seq_samp_hill / nb_launched)

        size += size_step

    data = {'misere': results_misere,
            'SeqSampHill': results_seq_samp_hill, 'beamSearch-30': results_beam,
            'beamSearch-50': results_beam_50,
            'beamSearch-100': results_beam_100}

    df = pd.DataFrame(data=data, index=x_axis)

    plt.clf()
    ax = sns.lineplot(data=df)

    ax.set(xlabel='Length max', ylabel='Average WRAcc top-10 patterns')

    with open('./space_size/result', 'w+') as f:
        f.write(' '.join([str(i) for i in x_axis]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_beam]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_seq_samp_hill]))

    plt.legend(loc='lower left')
    plt.savefig('./space_size/wracc_over_complexity.png')

    # plt.show()


def vertical_vs_horizontal():
    time_xp = 180

    vertical_results = []
    horizontal_results = []

    for data, target, enable_i in datasets:
        items = extract_items(data)

        average_vertical = average_results(
            seq_samp_hill(data, items, time_xp, target, 5, enable_i, False))
        average_horizontal = average_results(
            seq_samp_hill(data, items, time_xp, target, 5, enable_i, True))

        vertical_results.append(average_vertical)
        horizontal_results.append(average_horizontal)

    data = {'WRAcc': vertical_results + horizontal_results,
            'dataset': datasets_names + datasets_names,
            'Version': ['Gener_Speci' for i in range(len(vertical_results))] +
                       ['Mutation' for i in range(len(horizontal_results))]}

    df = pd.DataFrame(data=data)

    plt.clf()
    sns.barplot(x='dataset', y='WRAcc', hue='Version', data=df)
    plt.savefig('./vertical/vertical_vs_horizontal.png')

    with open('./vertical/result', 'w+') as f:
        f.write(' '.join([str(i) for i in vertical_results]))
        f.write('\n')
        f.write(' '.join([str(i) for i in horizontal_results]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    # plt.show()


def naive_vs_bitset():
    time_xp = 180

    naive_results = []
    bitset_results = []

    for data, target, enable_i in datasets:
        items = extract_items(data)

        average_naive = average_results(
            seq_samp_hill(data, items, time_xp, target, 10, enable_i, wracc_vertical=False))
        average_bitset = average_results(
            seq_samp_hill(data, items, time_xp, target, 10, enable_i, wracc_vertical=True))

        naive_results.append(average_naive)
        bitset_results.append(average_bitset)

    data = {'WRAcc': naive_results + bitset_results,
            'dataset': datasets_names + datasets_names,
            'Version': ['Naive' for i in range(len(naive_results))] +
                       ['Bitset' for i in range(len(bitset_results))]}

    df = pd.DataFrame(data=data)

    ax = sns.barplot(x='dataset', y='WRAcc', hue='Version', data=df)
    fig = ax.get_figure()

    fig.savefig('./naive/naive.png')

    with open('./naive/result', 'w+') as f:
        f.write(' '.join([str(i) for i in naive_results]))
        f.write('\n')
        f.write(' '.join([str(i) for i in bitset_results]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    # plt.show()


compare_datasets_seeds()
# compare_competitors()
# vertical_vs_horizontal()
# naive_vs_bitset()
# show_quality_over_time()
# compare_datasets()
# compare_ground_truth()
# quality_over_dataset_size()
