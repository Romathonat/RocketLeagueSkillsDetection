from multiprocessing.pool import Pool

import sys

import pandas as pd

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from competitors.beam_search import beam_search
from competitors.misere import misere
from competitors.misere_final_opti import misere_final_opti
from seqsamphill.seq_samp_hill import seq_samp_hill

from seqsamphill.utils import read_data, extract_items, \
    encode_items, \
    encode_data, print_results, average_results, read_data_kosarak, \
    read_data_sc2, reduce_k_length

from competitors.flat_UCB_optimized import flat_UCB_optimized
from competitors.flat_UCB import flat_UCB

from mctsextend.mctsextend_main import launch_mcts

sys.setrecursionlimit(500000)

# third element: enable_i
datasets = [
    (read_data_kosarak('../data/aslbu.data'), '195', False),
    (read_data('../data/promoters.data'), '+', False),
    (read_data_kosarak('../data/blocks.data'), '7', False),
    (read_data_kosarak('../data/context.data'), '4', False),
    (read_data('../data/splice.data'), 'EI', False),
    (read_data_sc2('../data/sequences-TZ-45.txt')[:5000], '1', False),
    (read_data_kosarak('../data/skating.data'), '1', False)
]

datasets_names = ['aslbu', 'promoters', 'blocks', 'context', 'splice', 'sc2', 'skating']


def compare_seeds(number_dataset):
    DATA = datasets[number_dataset][0]
    items = extract_items(DATA)
    target_class = datasets[number_dataset][1]
    enable_i = datasets[number_dataset][2]

    TIME = 20

    print('Dataset: {}'.format(datasets_names[number_dataset]))
    seq_samp_hill_results = misere(DATA, TIME, target_class, 5)
    print_results(seq_samp_hill_results)

    seed_results = flat_UCB_optimized(DATA, items, TIME, target_class, top_k=5, enable_i=enable_i, vertical=False)
    print_results(seed_results)

# compare_seeds(0)
# compare_seeds(1)
# compare_seeds(2)
# compare_seeds(3)
# compare_seeds(4)
# compare_seeds(5)
# compare_seeds(6)

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


def compare_datasets_UCB():
    pool = Pool(processes=3)
    time_xp = 5
    top_k = 10
    xp_repeat = 5

    misere_hist = []
    beam_hist = []
    misere_opti = []
    ucb_hist = []
    ucb_opti_hist = []

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        items = extract_items(data)

        sum_misere = 0
        sum_beam = 0
        sum_misere_opti = 0
        sum_UCB = 0
        sum_ucb_opti = 0

        for j in range(xp_repeat):
            result_misere = pool.apply_async(misere, (data, time_xp, target, top_k))
            result_beam = pool.apply_async(beam_search,
                                           (
                                               data, items, time_xp, target,
                                               enable_i, top_k))

            results_misere_opti = pool.apply_async(misere_final_opti, (
                data, items, time_xp, target, top_k))

            results_UCB = pool.apply_async(flat_UCB, (data, items, time_xp, target, top_k, enable_i))

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k, enable_i))

            result_misere = result_misere.get()
            result_beam = result_beam.get()
            results_misere_opti = results_misere_opti.get()
            results_UCB = results_UCB.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(result_misere) < top_k:
                print(
                    "Too few example on misere on dataset {}: {} results".format(datasets_names[i], len(result_misere)))

            if len(results_misere_opti) < top_k:
                print("Too few example on misere opti on dataset {}: {} results".format(datasets_names[i],
                                                                                        len(results_misere_opti)))

            if len(result_beam) < top_k:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(result_beam)))

            if len(result_ucb_opti) < top_k:
                print(
                    "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))

            if len(results_UCB) < top_k:
                print(
                    "Too few examples on ucb on dataset {}: {} results".format(datasets_names[i], len(result_ucb_opti)))

            sum_misere += average_results(result_misere)
            sum_beam += average_results(result_beam)
            sum_misere_opti += average_results(results_misere_opti)
            sum_UCB += average_results(results_UCB)
            sum_ucb_opti += average_results(result_ucb_opti)

        misere_hist.append(max(0, sum_misere / xp_repeat))
        beam_hist.append(max(0, sum_beam / xp_repeat))
        misere_opti.append(max(0, sum_misere_opti / xp_repeat))
        ucb_hist.append(max(0, sum_UCB / xp_repeat))
        ucb_opti_hist.append(max(0, sum_ucb_opti / xp_repeat))

    data = {'Mean WRAcc': beam_hist + misere_hist + misere_opti + ucb_hist + ucb_opti_hist,
            'dataset': datasets_names + datasets_names + datasets_names + datasets_names + datasets_names,
            'Algorithm':
                ['beam_search' for i in range(len(beam_hist))] +
                ['misere' for i in range(len(misere_hist))] +
                ['misere optimized' for i in range(len(misere_opti))] +
                ['UCB' for i in range(len(ucb_hist))] +
                ['UCB optimized' for i in range(len(ucb_opti_hist))]}

    df = pd.DataFrame(data=data)

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/barplot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        f.write(' '.join([str(i) for i in misere_opti]))
        f.write('\n')
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in ucb_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in ucb_opti_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    # plt.show()


def data_add(data, wracc, dataset, algorithm):
    data['WRAcc'].append(wracc)
    data['dataset'].append(dataset)
    data["Algorithm"].append(algorithm)


def violin_plot_datasets():
    pool = Pool(processes=5)
    time_xp = 20
    top_k = 5
    xp_repeat = 5

    misere_hist = []
    beam_hist = []
    misere_opti = []
    ucb_hist = []
    ucb_opti_hist = []

    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        items = extract_items(data)

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, time_xp, target, top_k))
            results_beam = pool.apply_async(beam_search,
                                            (
                                                data, items, time_xp, target,
                                                enable_i, top_k))

            # results_misere_opti = pool.apply_async(misere_final_opti, (
            #     data, items, time_xp, target, top_k))
            #
            # results_UCB = pool.apply_async(flat_UCB, (data, items, time_xp, target, top_k, enable_i))

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k, enable_i))

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            # results_misere_opti = results_misere_opti.get()
            # results_UCB = results_UCB.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_misere) < top_k:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))

            # if len(results_misere_opti) < top_k:
            #     print("Too few example on misere opti on dataset {}: {} results".format(datasets_names[i],
            #                                                                             len(results_misere_opti)))

            if len(results_beam) < top_k:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))

            if len(result_ucb_opti) < top_k:
                print(
                    "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))
            #
            # if len(results_UCB) < top_k:
            #     print(
            #         "Too few examples on ucb on dataset {}: {} results".format(datasets_names[i], len(result_ucb_opti)))

            data_add(data_final, max(0, average_results(results_misere)), datasets_names[i], 'misere')
            data_add(data_final, max(0, average_results(results_beam)), datasets_names[i], 'beam')
            # data_add(data_final, max(0, average_results(results_misere_opti)), datasets_names[i], 'misere optimized')
            # data_add(data_final, max(0, average_results(results_UCB)), datasets_names[i], 'UCB')
            data_add(data_final, max(0, average_results(result_ucb_opti)), datasets_names[i], 'UCB optimized')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/violin_plot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        # f.write(' '.join([str(i) for i in misere_opti]))
        # f.write('\n')
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        # f.write(' '.join([str(i) for i in ucb_hist]))
        # f.write('\n')
        f.write(' '.join([str(i) for i in ucb_opti_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    plt.show()

def average_results_normalize_wracc(results):
    normalized_results = []
    for result in results:
        #normalized_results.append(((result[0] * 4), result[1]))
        normalized_results.append(((result[0]), result[1]))
    return average_results(normalized_results)


def boxplot_dataset_iterations():
    pool = Pool(processes=5)
    time_xp = 10000000000000
    iterations_limit = 1000
    top_k = 5
    xp_repeat = 5

    misere_hist = []
    beam_hist = []
    ucb_opti_hist = []

    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        items = extract_items(data)

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, time_xp, target),
                                              {'top_k': top_k, 'iterations_limit': iterations_limit})
            results_beam = pool.apply_async(beam_search,
                                            (data, items, time_xp, target),
                                            {'enable_i': enable_i, 'top_k': top_k,
                                             'iterations_limit': iterations_limit})

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k),
                                               {'enable_i': enable_i, 'iterations_limit': iterations_limit})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_misere) < top_k:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))

            if len(results_beam) < top_k:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))

            if len(result_ucb_opti) < top_k:
                print(
                    "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))

            data_add(data_final, max(0, average_results_normalize_wracc(results_misere)), datasets_names[i], 'misere')
            data_add(data_final, max(0, average_results_normalize_wracc(results_beam)), datasets_names[i], 'beam')
            data_add(data_final, max(0, average_results_normalize_wracc(result_ucb_opti)), datasets_names[i], 'SeqScout')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/iterations_boxplot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in ucb_opti_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    plt.show()


def data_add_over_iterations(data, wracc, iterations, algorithm):
    data['WRAcc'].append(wracc)
    data['iterations'].append(iterations)
    data['Algorithm'].append(algorithm)


def show_quality_over_iterations_ucb():
    number_dataset = 5
    data, target, enable_i = datasets[number_dataset]

    items = extract_items(data)
    time_xp = 10000000000000
    iterations_limit = 50

    pool = Pool(processes=5)

    # if we want to average
    nb_launched = 5

    top_k = 5

    iterations_step = 200

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}

    for i in range(11):
        print('Iteration: {}'.format(i))

        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, time_xp, target),
                                              {'top_k': top_k, 'iterations_limit': iterations_limit})
            results_beam = pool.apply_async(beam_search,
                                            (data, items, time_xp, target),
                                            {'enable_i': enable_i, 'top_k': top_k,
                                             'iterations_limit': iterations_limit})

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k),
                                               {'enable_i': enable_i, 'iterations_limit': iterations_limit})


            data_add_over_iterations(data_final, max(0, average_results_normalize_wracc(results_misere.get())), iterations_limit, 'misere')
            data_add_over_iterations(data_final, max(0, average_results_normalize_wracc(results_beam.get())), iterations_limit, 'beam')
            data_add_over_iterations(data_final, max(0, average_results_normalize_wracc(result_ucb_opti.get())), iterations_limit, 'SeqScout')


        iterations_limit += iterations_step


    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')

    #ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./time_ucb/over_iterations.png')

    # with open('./time_ucb/result', 'w+') as f:
    #     f.write(' '.join([str(i) for i in x_axis]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_beam]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_misere]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_ucb_opti]))

    plt.show()


def mcts_boxplot_datasets():
    pool = Pool(processes=5)
    time_xp = 5
    top_k = 10
    xp_repeat = 1

    misere_hist = []
    beam_hist = []
    ucb_opti_hist = []
    mcts_hist = []

    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        items = extract_items(data)

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, time_xp, target, top_k))
            results_beam = pool.apply_async(beam_search,
                                            (
                                                data, items, time_xp, target,
                                                enable_i, top_k))
            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k, enable_i))
            results_mcts = pool.apply_async(launch_mcts, (data, time_xp, target, top_k))

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()
            results_mcts = results_mcts.get()

            if len(results_misere) < top_k:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < top_k:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))

            if len(result_ucb_opti) < top_k:
                print(
                    "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))

            if len(result_ucb_opti) < top_k:
                print(
                    "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_mcts)))

            data_add(data_final, max(0, average_results(results_misere)), datasets_names[i], 'misere')
            data_add(data_final, max(0, average_results(results_beam)), datasets_names[i], 'beam')
            data_add(data_final, max(0, average_results(result_ucb_opti)), datasets_names[i], 'UCB optimized')
            data_add(data_final, max(0, average_results(results_mcts)), datasets_names[i], 'surprise')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)
    plt.savefig('./wracc_datasets/violin_plot.png')

    with open('./wracc_datasets/result', 'w+') as f:
        f.write(' '.join([str(i) for i in misere_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in mcts_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in beam_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in ucb_opti_hist]))
        f.write('\n')
        f.write(' '.join([str(i) for i in datasets_names]))

    plt.show()


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


def show_quality_over_time_ucb():
    DATA = read_data_kosarak('../data/skating.data')
    # DATA = read_data('../data/promoters.data')
    # DATA = read_data('../data/splice.data')
    # DATA = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]

    items = extract_items(DATA)
    # items, item_to_encoding, encoding_to_item = encode_items(items)
    # DATA = encode_data(DATA, item_to_encoding)
    target_class = '1'

    pool = Pool(processes=5)

    # if we want to average
    nb_launched = 10

    results_beam = []
    results_misere = []
    results_misere_opti = []
    results_ucb = []
    results_ucb_opti = []

    x_axis = []

    top_k = 5

    # time_step = 17
    time_step = 5

    xp_time = 5

    for i in range(11):
        print('Iteration: {}'.format(i))
        x_axis.append(xp_time)

        average_nb_launched_beam = 0
        average_nb_launched_misere = 0
        average_nb_launched_misere_opti = 0
        average_nb_launched_ucb = 0
        average_nb_launched_ucb_opti = 0

        for i in range(nb_launched):
            result_misere = pool.apply_async(misere,
                                             (DATA, xp_time, target_class, top_k))

            result_beam = pool.apply_async(beam_search,
                                           (DATA, items, xp_time, target_class,
                                            False, top_k, 30))

            result_misere_opti = pool.apply_async(misere_final_opti, (
                DATA, items, xp_time, target_class, top_k))

            result_ucb = pool.apply_async(flat_UCB, (DATA, items, xp_time, target_class, top_k, True))

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (DATA, items, xp_time, target_class, top_k, True))

            average_nb_launched_misere += average_results(result_misere.get())
            average_nb_launched_beam += average_results(result_beam.get())
            average_nb_launched_misere_opti += average_results(result_misere_opti.get())
            average_nb_launched_ucb += average_results(result_ucb.get())
            average_nb_launched_ucb_opti += average_results(result_ucb_opti.get())

        results_beam.append(average_nb_launched_beam / nb_launched)
        results_misere.append(average_nb_launched_misere / nb_launched)
        results_misere_opti.append(average_nb_launched_misere_opti / nb_launched)
        results_ucb.append(average_nb_launched_ucb / nb_launched)
        results_ucb_opti.append(average_nb_launched_ucb_opti / nb_launched)

        xp_time += time_step

    data = {'beam-search': results_beam,
            'misere': results_misere,
            'misere optimized': results_misere_opti,
            'UCB': results_ucb,
            'UCB optimized': results_ucb_opti}

    df = pd.DataFrame(data=data, index=x_axis)

    plt.clf()
    ax = sns.lineplot(data=df)

    # ax = sns.lineplot(x='x', y=('misereHill', 'misere'), data=df)

    ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./time_ucb/aslbu_over_time.png')

    with open('./time_ucb/result', 'w+') as f:
        f.write(' '.join([str(i) for i in x_axis]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_beam]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_misere_opti]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_ucb]))
        f.write('\n')
        f.write(' '.join([str(i) for i in results_ucb_opti]))

    plt.show()


def compare_ground_truth():
    data = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    data = reduce_k_length(10, data)

    items = extract_items(data)

    target = '1'
    enable_i = True
    time_xp = 100000000000

    pool = Pool(processes=3)

    top_k = 10
    iterations_limit = 50
    iteration_step = 1000

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}
    # found with exaustive search
    ground_truth = 0.008893952000000009

    for i in range(10):
        print('Iteration: {}'.format(i))
        # if we want to average
        nb_launched = 5

        for i in range(nb_launched):
            # results_misere = pool.apply_async(misere, (data, time_xp, target),
            #                                   {'top_k': top_k, 'iterations_limit': iterations_limit})
            # results_beam = pool.apply_async(beam_search,
            #                                 (data, items, time_xp, target),
            #                                 {'enable_i': enable_i, 'top_k': top_k,
            #                                  'iterations_limit': iterations_limit})

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k),
                                               {'enable_i': enable_i, 'iterations_limit': iterations_limit})

            # data_add_over_iterations(data_final, max(0, average_results(results_misere.get())) / ground_truth, iterations_limit, 'misere')
            # data_add_over_iterations(data_final, max(0, average_results(results_beam.get())) / ground_truth, iterations_limit, 'beam')
            data_add_over_iterations(data_final, max(0, average_results(result_ucb_opti.get())) / ground_truth, iterations_limit, 'SeqScout')

        iterations_limit += iteration_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')

    ax.set(xlabel='iterations', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./ground_truth/gt.png')

    df.to_pickle('./ground_truth/result')

    plt.show()


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
    time_xp = 5

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

    plt.show()


def data_add_over_theta(data, wracc, theta, algorithm):
    data['WRAcc'].append(wracc)
    data['theta'].append(theta)
    data['Algorithm'].append(algorithm)


def quality_over_theta():
    number_dataset = 6
    data, target, enable_i = datasets[number_dataset]

    items = extract_items(data)
    time_xp = 10000000000000
    iterations_limit = 1000

    pool = Pool(processes=5)

    # if we want to average
    nb_launched = 5
    top_k = 5

    theta = 0

    data_final = {'WRAcc': [], 'theta': [], 'Algorithm': []}

    for i in range(11):
        print('Iteration: {}'.format(i))
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, time_xp, target),
                                              {'top_k': top_k, 'iterations_limit': iterations_limit, 'theta': theta})
            results_beam = pool.apply_async(beam_search,
                                            (data, items, time_xp, target),
                                            {'enable_i': enable_i, 'top_k': top_k,
                                             'iterations_limit': iterations_limit, 'theta': theta})

            result_ucb_opti = pool.apply_async(flat_UCB_optimized, (data, items, time_xp, target, top_k),
                                               {'enable_i': enable_i, 'iterations_limit': iterations_limit, 'theta': theta})


            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_beam ) < top_k:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < top_k:
                print("Too few seqscout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < top_k:
                print("Too few misere: {}".format(len(results_misere)))

            data_add_over_theta(data_final, max(0, average_results_normalize_wracc(results_misere)), theta, 'misere')
            data_add_over_theta(data_final, max(0, average_results_normalize_wracc(results_beam)), theta, 'beam')
            data_add_over_theta(data_final, max(0, average_results_normalize_wracc(result_ucb_opti)), theta, 'SeqScout')

        theta += 0.1


    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='theta', y='WRAcc', hue='Algorithm')

    #ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./theta/over_theta.png')

    df.to_pickle('./theta/result')

    # with open('./time_ucb/result', 'w+') as f:
    #     f.write(' '.join([str(i) for i in x_axis]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_beam]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_misere]))
    #     f.write('\n')
    #     f.write(' '.join([str(i) for i in results_ucb_opti]))

    plt.show()



# compare_datasets_UCB()
# violin_plot_datasets()
boxplot_dataset_iterations()
# quality_over_theta()
# show_quality_over_iterations_ucb()
# mcts_boxplot_datasets()
# compare_competitors()
# vertical_vs_horizontal()
# naive_vs_bitset()
# show_quality_over_time()
# show_quality_over_time_ucb()
# compare_datasets()
# compare_ground_truth()
# quality_over_dataset_size()
