
def read_jmlr(target_word):
    return_sequences = []

    with open('./jmlr.lab') as dict_files:
        data_dict = {}
        for i, line in enumerate(dict_files):
            data_dict[i] = line[:-1]

    with open('./jmlr.dat') as jmlr:
        data = jmlr.readline()
        data = data.split("-1")

        for seq in data:
            sequence = [set([data_dict[int(i)]]) for i in seq.split(" ") if i != '']
            return_sequences.append(sequence)

    # now we add the class for the presence of a word, and we remove the target word
    for sequence in return_sequences:
        if set([target_word]) in sequence:
            sequence.insert(0, '+')

            # now we remove the element !
            while "Removing the element":
                try:
                    sequence.remove(set([target_word]))
                except ValueError:
                   break
        else:
            sequence.insert(0, '-')

    return return_sequences

if __name__ == '__main__':
    data = read_jmlr('machin')

    for dat in data:
        print(dat)
