from seqsamphill.utils import read_data, read_data_kosarak, read_data_sc2,k_length, extract_items

datasets = [
        (read_data_kosarak('../data/aslbu.data'), 'aslbu'),
        (read_data('../data/promoters.data'), 'promoters'),
        (read_data('../data/splice.data'), 'splice'),
        (read_data_kosarak('../data/blocks.data'), 'blocks'),
        (read_data_kosarak('../data/context.data'), 'context'),
        (read_data_sc2('../data/sequences-TZ-45.txt')[:5000], 'sc2'),
        (read_data_kosarak('../data/skating.data'), 'skating')
]


for dataset, name in datasets:
    for i in range(len(dataset)):
        dataset[i] = dataset[i][1:]

    k_max = 0
    n_max = 0

    for line in dataset:
        if k_length(line) > k_max:
           k_max = k_length(line)

        if len(line) > n_max:
            n_max = len(line)

    print('k_max: {}'.format(k_max))
    print('n_max: {}'.format(n_max))
    print('m: {}'.format(len(extract_items(dataset))))
    print('Lines number : {}'.format(len(dataset)))


