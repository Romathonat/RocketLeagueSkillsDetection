import json
import copy

figures = {
    1: 'Ceiling_shot',
    2: 'PowerShot',
    3: 'Wawing_Dash',
    4: 'Pinch_Shot',
    5: 'Air_Dribling',
    6: 'Flick',
    7: 'Musty_Flick',
    8: 'Double_Tap',
    -1: 'Fail'
}

dict_histo = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    -1: 0
}

with open('../data/rocket_league_new.json', 'r') as f:
    data = json.load(f)
    previous_figures = []

    for figure in data:
        # if we do not have any error

        if len(figure['sequence']) < 130:
            dict_histo[int(figure['figure'])] += 1
        '''
        # cleaning
        if figure not in previous_figures:
            previous_figures.append(copy.deepcopy(figure))
        else: print("duplicate found")
        '''

    print("The number of skills is {}".format(sum([value for _, value in dict_histo.items()])))
    print({figures[key]: value for key, value in dict_histo.items()})

