import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from seqscout.seq_scout import stratified_k_fold, seq_scout
from seqscout.utils import read_json_rl
from seqscout.seq_scout import preprocess

SHOW = True

def quality_over_pattern_number():
    # maybe recode to speed up ? Use 120 rules from seqscout then launch DT for subset of this 120 -> need to store all rules for all classes for all train/test
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
    data_final = {"alpha": [], "Pattern Mean WRAcc": []}

    for alpha in [x * 0.1 for x in range(0, 10)]:
        DATA = read_json_rl('../data/final_sequences.json')

        _, accuracy_list = stratified_k_fold()

        # X = np.array([i[1:] for i in DATA])
        # Y = np.array([[i[0]] for i in DATA])

        numerics_values = preprocess(DATA)

        mean_wracc = seq_scout(DATA, "1", numerics_values=numerics_values, numeric_remove_proba=alpha)

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


if __name__ == '__main__':
    # quality_over_pattern_number()
    pattern_WRAcc_vs_alpha()

