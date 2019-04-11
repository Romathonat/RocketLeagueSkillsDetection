import pandas as pd

from sax.normalize import normalize
from sax.paa import paa
from sax.lookup_table import lookup_table

# TODO: denormalize to have the true value !


def check_position_lookup_table(lookup, value):
    '''
    :param lookup:
    :param value:
    :return: returns the closest inferior breakpoint to value, or -inf if we are in the case [-inf, breakpoint1]
    '''

    if value < lookup[0]:
        return -float('inf')

    if value > lookup[-1]:
        return len(lookup) - 1

    for i in range(len(lookup) - 1):
        if value > lookup[i] and value < lookup[i + 1]:
            return i

def convert_position_to_threshold(lookup, position_symbol):
    '''
    :param lookup: lookup table
    :param position_symbol: the symbol given by check_position_lookup_table
    :return: the interval corresponding to symbol
    '''
    if position_symbol == -float('inf'):
        return '[-inf, {}]'.format(lookup[0])

    elif position_symbol == len(lookup):
        return '[{}, -inf]'.format(lookup[position_symbol])

    else:
        return '[{}, {}]'.format(lookup[position_symbol], lookup[position_symbol + 1])

def readable_pattern(pattern, lookup, original_mean, original_std):
    '''
    :param pattern: the pattern we want to translate
    :param lookup: lookup table
    :param original_mean: original mean to denormalize
    :param original_std: original std to denormalize
    :return: the readable pattern
    '''
    pass


def sax(ts, w, a):
    '''
    :param ts: the timeserie on which we want to apply sax
    :param w: the number of slot for the paa
    :param a: the number of items of the language (the more we have the more precise we are)
    :return: the sax representation, with numbers: 1 means the value is between 1 and 2 in lookup table
    '''
    lookup = lookup_table(a)

    ts = normalize(ts)
    ts_paa = paa(ts, w)

    sequence = []

    for _, elt in enumerate(ts_paa):
        sequence.append(check_position_lookup_table(lookup, elt))

    return pd.Series(sequence, index=ts_paa.index)


def sax_slot_size(ts, slot_size, a):
    '''
    :param ts: the time serie
    :param slot_size: the slot size to average elements (< len(ts))
    :param a: the number of items of the language
    :return:
    '''
    w = len(ts) // slot_size
    return sax(ts, w, a)

