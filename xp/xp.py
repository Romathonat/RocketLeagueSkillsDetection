import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from seqscout.seq_scout import stratified_k_fold, seq_scout
from seqscout.utils import read_json_rl
from seqscout.seq_scout import preprocess
from seqscout.dtw import launch_knn_dtw

SHOW = False


def quality_over_pattern_number():
    pattern_numbers = [3, 5, 10, 20, 30, 50, 100, 120]
    data_final = {"Pattern number": [], "Accuracy": []}

    for pattern_number in pattern_numbers:
        _, accuracy_list = stratified_k_fold(pattern_number=pattern_number)
        for data_point in accuracy_list:
            data_final["Pattern number"].append(pattern_number)
            data_final["Accuracy"].append(data_point)

    df = pd.DataFrame(data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='Pattern number', y='Accuracy')

    plt.savefig('./quality_over_pattern_number/qual_vs_pattern_number.png')
    df.to_pickle('./quality_over_pattern_number/result')

    if SHOW:
        plt.show()


def pattern_WRAcc_vs_alpha():
    data_final = {"alpha": [], "Patterns Mean WRAcc": []}

    for alpha in [x * 0.1 for x in range(0, 11)]:
        # in order to have have bar errors
        for _ in range(5):
            DATA = read_json_rl('../data/rocket_league_new.json')

            numerics_values = preprocess(DATA)

            patterns = seq_scout(DATA, "1", numerics_values=numerics_values, numeric_remove_proba=alpha)

            sum_result = 0
            for result in patterns:
                sum_result += result[0]
            mean_wracc = sum_result / len(patterns)

            data_final["alpha"].append(alpha)
            data_final["Patterns Mean WRAcc"].append(mean_wracc)

    df = pd.DataFrame(data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='alpha', y='Patterns Mean WRAcc')

    plt.savefig('./wracc_vs_alpha/wracc_vs_alpha.png')
    df.to_pickle('./wracc_vs_alpha/result')

    if SHOW:
        plt.show()


def accuracy_vs_iteration_number():
    iteration_numbers = [500, 1000, 2000, 4000, 8000, 10000]
    data_final = {"Iteration number": [], "Accuracy": []}

    for iteration_number in iteration_numbers:
        _, accuracy_list = stratified_k_fold(iteration_limit=iteration_number)
        for data_point in accuracy_list:
            data_final["Iteration number"].append(iteration_number)
            data_final["Accuracy"].append(data_point)

    df = pd.DataFrame(data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='Iteration number', y='Accuracy')

    plt.savefig('./qual_vs_iter/qual_vs_iter.png')
    df.to_pickle('./qual_vs_iter/result')

    if SHOW:
        plt.show()


def accuracy_vs_diversity():
    data_final = {"Theta": [], "Accuracy": []}

    for theta in [x * 0.1 for x in range(0, 11)]:
        _, accuracy_list = stratified_k_fold(theta=theta)

        for data_point in accuracy_list:
            data_final["Theta"].append(theta)
            data_final["Accuracy"].append(data_point)

    df = pd.DataFrame(data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.lineplot(data=df, x='Theta', y='Accuracy')

    plt.savefig('./qual_vs_theta/qual_vs_theta.png')
    df.to_pickle('./qual_vs_theta/result')

    if SHOW:
        plt.show()


def barplot_classif():
    algorithms_classif = ["DT", "RF", "XGB", "SVM", "NB"]
    data_final = {'Accuracy': [], 'Classifier': []}

    for algo in algorithms_classif:
        print("Algo: {}".format(algo))
        _, accuracy_list = stratified_k_fold(classif=algo, numeric_remove_proba=0.8, pattern_number=20,
                                             iteration_limit=10000, no_filtering=True)

        for data_point in accuracy_list:
            data_final["Classifier"].append(algo)
            data_final["Accuracy"].append(data_point)

    df = pd.DataFrame(data=data_final)

    sns.set(rc={'figure.figsize': (8, 6.5)})

    plt.clf()
    ax = sns.barplot(x='Classifier', y='Accuracy', data=df)

    plt.savefig('./barplot_classif/barplot_classif.png')
    df.to_pickle('./barplot_classif/result')

    if SHOW:
        plt.show()


def seqscout_vs_dtw():
    # we compute the time taken by algorithm
    start_time = time.time()
    mean_accuracy_seqscout, _ = stratified_k_fold(numeric_remove_proba=0.8, classif="RF", pattern_number=20,
                                                  iteration_limit=8000, theta=0.8, no_filtering=True)
    print("Our method took {}".format(time.time() - start_time))

    start_time = time.time()
    mean_accuracy_dtw = launch_knn_dtw()
    print(f"1-NN took {time.time() - start_time}")

    with open('./seqscout_vs_dtw/result', 'w') as f:
        f.write(str(mean_accuracy_seqscout))
        f.write(str(mean_accuracy_dtw))


if __name__ == '__main__':
    # quality_over_pattern_number()
    # pattern_WRAcc_vs_alpha()
    # accuracy_vs_iteration_number()
    # accuracy_vs_diversity()
    # barplot_classif()
    seqscout_vs_dtw()
