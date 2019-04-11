from scipy.stats import norm


def lookup_table(x):
    breaks = []
    slot_sum = 1 / x

    for i in range(1, x):
        breaks.append(norm.ppf(slot_sum * i))

    return breaks

