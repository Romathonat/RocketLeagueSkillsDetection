from multiprocessing.pool import Pool

import sys

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from competitors.beam_search import beam_search
from competitors.misere import misere

from seqscout.utils import read_data, extract_items, \
    print_results, average_results, read_data_kosarak, \
    read_data_sc2, reduce_k_length, k_length, read_jmlr
from seqscout.seq_scout import seq_scout
import seqscout.global_var
from seqscout.conf import TIME_BUDGET_XP, TOP_K

sys.setrecursionlimit(500000)

# third element: enable_i
datasets = [
    (read_data_kosarak('../data/aslbu.data'), '195', False),
    (read_data('../data/promoters.data'), '+', False),
    (read_data_kosarak('../data/blocks.data'), '7', False),
    (read_data_kosarak('../data/context.data'), '4', False),
    (read_data('../data/splice.data'), 'EI', False),
    (read_data_sc2('../data/sequences-TZ-45.txt')[:5000], '1', False),
    (read_data_kosarak('../data/skating.data'), '1', False),
    (read_jmlr('svm', '../data/jmlr/jmlr'), '+', False)
]

datasets_names = ['aslbu', 'promoters', 'blocks', 'context', 'splice', 'sc2', 'skating', 'jmlr']

SHOW = False


def data_add_generic(data, **kwargs):
    for key, value in kwargs.items():
        data[key].append(value)


def average_results_wracc(results):
    normalized_results = []
    for result in results:
        normalized_results.append(((result[0]), result[1]))
    return average_results(normalized_results)


def boxplot_dataset_iterations():
    pool = Pool(processes=5)
    xp_repeat = 5

    data_final = {'WRAcc': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))

        for j in range(xp_repeat):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': TIME_BUDGET_XP})
            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_misere) < TOP_K:
                print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(results_misere)))
            if len(results_beam) < TOP_K:
                print(
                    "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                      len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print(
                    "Too few example on SeqScout on dataset {}: {} results".format(datasets_names[i],
                                                                                   len(result_ucb_opti)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), dataset=datasets_names[i],
                             Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), dataset=datasets_names[i],
                             Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), dataset=datasets_names[i],
                             Algorithm='SeqScout')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='dataset', y='WRAcc', hue='Algorithm', data=df)

    plt.savefig('./wracc_datasets/iterations_boxplot.png')
    df.to_pickle('./wracc_datasets/result')

    if SHOW:
        plt.show()


def show_quality_over_iterations_ucb(number_dataset):
    data, target, enable_i = datasets[number_dataset]

    # if we want to average
    nb_launched = 5
    pool = Pool(processes=3)

    iterations_limit = 50
    iterations_step = 1000

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}

    for i in range(12):
        print('Iteration: {}'.format(i))

        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP, 'iterations_limit': iterations_limit})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                             'iterations_limit': iterations_limit})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'iterations_limit': iterations_limit})

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere.get())), iterations=iterations_limit,
                             Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam.get())), iterations=iterations_limit,
                             Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti.get())), iterations=iterations_limit,
                             Algorithm='SeqScout')

        iterations_limit += iterations_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')

    plt.savefig('./time_ucb/over_iterations{}.png'.format(datasets_names[number_dataset]))
    df.to_pickle('./time_ucb/result')

    if SHOW:
        plt.show()


def compare_ground_truth():
    data = read_data_sc2('../data/sequences-TZ-45.txt')[:5000]
    data = reduce_k_length(10, data)

    target = '1'
    enable_i = True

    # if we want to average
    nb_launched = 5
    pool = Pool(processes=3)

    iterations_limit = 50
    iteration_step = 1000

    data_final = {'WRAcc': [], 'iterations': [], 'Algorithm': []}

    # found with exaustive search
    ground_truth = 0.008893952000000009

    for i in range(10):
        print('Iteration: {}'.format(i))

        for i in range(nb_launched):
            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'iterations_limit': iterations_limit})

            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti.get())) / ground_truth,
                             iterations=iterations_limit, Algorithm='SeqScout')

        iterations_limit += iteration_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})
    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='WRAcc', hue='Algorithm')
    ax.set(xlabel='iterations', ylabel='WRAcc')
    plt.savefig('./ground_truth/gt.png')
    df.to_pickle('./ground_truth/result')

    if SHOW:
        plt.show()


def naive_vs_bitset_seqscout():
    time_xp = 18

    for i, (data, target, enable_i) in enumerate(datasets):
        # we reset the count of iterations
        items = extract_items(data)

        seq_scout(data, items, target, time_xp, 10, enable_i, vertical=False)

        seq_scout(data, items, target, time_xp, 10, enable_i, vertical=True)

    # we need to look at the console output


def quality_over_theta():
    number_dataset = 1
    data, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=5)

    # if we want to average
    nb_launched = 5

    theta = 0.1

    data_final = {'WRAcc': [], 'theta': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP, 'theta': theta})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP, 'theta': theta})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP,
                                                'theta': theta})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_beam) < TOP_K:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print("Too few seqscout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < TOP_K:
                print("Too few misere: {}".format(len(results_misere)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), theta=theta, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), theta=theta, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), theta=theta,
                             Algorithm='SeqScout')

        theta += 0.1

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='theta', y='WRAcc', hue='Algorithm')

    plt.savefig('./theta/over_theta.png')

    df.to_pickle('./theta/result')

    if SHOW:
        plt.show()


def quality_over_top_k():
    number_dataset = 3
    data, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=3)

    # if we want to average
    nb_launched = 5
    top_k = 1

    data_final = {'WRAcc': [], 'top_k': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'top_k': top_k, 'time_budget': TIME_BUDGET_XP})
            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i, 'top_k': top_k,
                                             'time_budget': TIME_BUDGET_XP})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'top_k': top_k,
                                                'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_beam) < top_k:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < top_k:
                print("Too few seqscout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < top_k:
                print("Too few misere: {}".format(len(results_misere)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), top_k=top_k, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), top_k=top_k, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), top_k=top_k,
                             Algorithm='SeqScout')

        top_k += 10

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='top_k', y='WRAcc', hue='Algorithm')

    # ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./top_k/over_top_k.png')
    df.to_pickle('./top_k/result')

    if SHOW:
        plt.show()


def quality_over_size():
    number_dataset = 1
    data_origin, target, enable_i = datasets[number_dataset]

    pool = Pool(processes=3)

    # if we want to average
    nb_launched = 5

    size = 15
    size_step = 4
    data_final = {'WRAcc': [], 'size': [], 'Algorithm': []}

    for i in range(10):
        print('Iteration: {}'.format(i))
        data = reduce_k_length(size, data_origin)
        for i in range(nb_launched):
            results_misere = pool.apply_async(misere, (data, target),
                                              {'time_budget': TIME_BUDGET_XP})

            results_beam = pool.apply_async(beam_search,
                                            (data, target),
                                            {'enable_i': enable_i,
                                             'time_budget': TIME_BUDGET_XP})

            result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                               {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

            results_misere = results_misere.get()
            results_beam = results_beam.get()
            result_ucb_opti = result_ucb_opti.get()

            if len(results_beam) < TOP_K:
                print("Too few beam: {}".format(len(results_beam)))
            if len(result_ucb_opti) < TOP_K:
                print("Too few seqscout: {}".format(len(result_ucb_opti)))
            if len(results_misere) < TOP_K:
                print("Too few misere: {}".format(len(results_misere)))

            data_add_generic(data_final, WRAcc=max(0, average_results(results_misere)), size=size, Algorithm='misere')
            data_add_generic(data_final, WRAcc=max(0, average_results(results_beam)), size=size, Algorithm='beam')
            data_add_generic(data_final, WRAcc=max(0, average_results(result_ucb_opti)), size=size,
                             Algorithm='SeqScout')

        size += size_step

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='size', y='WRAcc', hue='Algorithm')
    ax.set(xlabel='Length max', ylabel='WRAcc')

    # ax.set(xlabel='Time(s)', ylabel='Average WRAcc top-10 patterns')

    plt.savefig('./space_size/over_size.png')
    df.to_pickle('./space_size/result')

    if SHOW:
        plt.show()


def number_iterations_optima():
    iterations_limit = 1000

    # if we want to average
    nb_launched = 5

    data_final = {'cost': [], 'iterations': [], 'dataset_name': []}
    for j, (data, target, enable_i) in enumerate(datasets):
        for i in range(nb_launched):
            # we reset the count of iterations
            seqscout.global_var.ITERATION_NUMBER = 0

            result_ucb_opti = seq_scout(data, target, enable_i=enable_i, time_budget=TIME_BUDGET_XP)

            if len(result_ucb_opti) < TOP_K:
                print("Too few seqscout: {}".format(len(result_ucb_opti)))

            iterations = 1000

            additional_iterations = seqscout.global_var.ITERATION_NUMBER - iterations_limit

            for i in range(10):
                data_add_generic(data_final, cost=additional_iterations / iterations, iterations=iterations,
                                 dataset_name=datasets_names[j])
                iterations += 2000

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='iterations', y='cost', hue='dataset_name')

    plt.savefig('./number_iterations_optima/it_optima.png')
    df.to_pickle('./number_iterations_optima/result')

    if SHOW:
        plt.show()


def add_lengths(patterns, dataset_name, data_final, algo):
    for pattern in patterns:
        k_length_p = k_length(pattern[1])
        data_add_generic(data_final, Length=k_length_p, dataset=dataset_name, Algorithm=algo)


def boxplots_description_lengths():
    pool = Pool(processes=3)

    data_final = {'Length': [], 'dataset': [], 'Algorithm': []}

    for i, (data, target, enable_i) in enumerate(datasets):
        print("Dataset {}".format(datasets_names[i]))
        items = extract_items(data)

        results_misere = pool.apply_async(misere, (data, target),
                                          {'time_budget': TIME_BUDGET_XP})
        results_beam = pool.apply_async(beam_search,
                                        (data, target),
                                        {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

        result_ucb_opti = pool.apply_async(seq_scout, (data, target),
                                           {'enable_i': enable_i, 'time_budget': TIME_BUDGET_XP})

        results_misere = results_misere.get()
        results_beam = results_beam.get()
        result_ucb_opti = result_ucb_opti.get()

        if len(results_misere) < TOP_K:
            print("Too few example on misere on dataset {}: {} results".format(datasets_names[i],
                                                                               len(results_misere)))

        if len(results_beam) < TOP_K:
            print(
                "Too few example on beam_search on dataset {}: {} results".format(datasets_names[i],
                                                                                  len(results_beam)))

        if len(result_ucb_opti) < TOP_K:
            print(
                "Too few example on flat UCB on dataset {}: {} results".format(datasets_names[i],
                                                                               len(result_ucb_opti)))

        add_lengths(results_misere, datasets_names[i], data_final, 'misere')
        add_lengths(results_beam, datasets_names[i], data_final, 'beam')
        add_lengths(result_ucb_opti, datasets_names[i], data_final, 'SeqScout')

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.boxplot(x='dataset', y='Length', hue='Algorithm', data=df)
    plt.savefig('./lengths/boxplot.png')
    df.to_pickle('./lengths/result')

    if SHOW:
        plt.show()


if __name__ == '__main__':
    #boxplot_dataset_iterations()
    #quality_over_theta()
    show_quality_over_iterations_ucb(1)
    show_quality_over_iterations_ucb(5)
    show_quality_over_iterations_ucb(6)
    #compare_ground_truth()
    #quality_over_top_k()
    #quality_over_size()
    #naive_vs_bitset_seqscout()
    #number_iterations_optima()
    #boxplots_description_lengths()
    #naive_vs_bitset_seqscout()
