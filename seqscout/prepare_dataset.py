dataset_names = ['aslbu', 'auslan2', 'skating', 'blocks', 'context']

for dataset in dataset_names:
    output = ''

    with open('../data/{}/{}_intervals.int'.format(dataset, dataset)) as itemsets:
        with open('../data/{}/{}_windows.int'.format(dataset, dataset)) as sequences:
            for sequence in sequences:
                class_s, begin_s, end_s = [int(i) for i in sequence.split('\t')]

                sequence_str = str(class_s)

                itemsets.seek(0)

                for item in itemsets:
                    attribute, begin, end = [int(i) for i in item.split('\t')]
                    if begin >= begin_s and end <= end_s:
                        sequence_str += ' {} -1'.format(str(attribute))
                    elif begin > end:
                        # useless to go further
                        break

                sequence_str += ' -2 \n'
                output += sequence_str

    with open('../data/{}.data'.format(dataset), 'w+') as out:
        out.write(output)

