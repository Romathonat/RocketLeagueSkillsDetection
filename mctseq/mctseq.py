# -*- coding: utf-8 -*-

"""Main module."""

ITEMS = set()


# Todo: command line interface, with pathfile of data

def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            sequence = [set(i.replace(' ', '')) for i in line[:-3].split('-1')]
            sequences.append(sequence)
    return sequences


def extract_items(data):
    items = set()
    for sequence in data:
        for itemset in sequence:
            for item in itemset:
                items.add(item)
    return items


data = read_data('/home/romain/Documents/contextPrefixSpan.txt')
items = extract_items(data)
