import pickle
from inputs import get_key, devices
import copy
import json

PATH_FINAL_OUTPUT = './replays/replay3/final_sequences.json'
with open('./replays/replay3/final3.dat', 'rb') as f:
    game_sequence = pickle.load(f)
    
'''
Figures:
Ceiling shot: 1
PowerShot: 2
Wawing Dash: 3
Pinch Shot: 4
Air Dribling: 5
Front Flick: 6
Musty Flick: 7
'''  

print('The first input is at {} At what time does the replay begins (in seconds)?'.format(game_sequence[0][1]['Time']))
replay_begin = input()

offset = game_sequence[0][1]['Time'] - float(replay_begin)
# need time begin, time end, figure name
sequences = []
while(True):
    print('Figure Number:')
    figure_number = input()
    
    print('Time begin (in seconds)')
    begin = float(input()) + offset
    
    print('Time end (in seconds)')
    end = float(input()) + offset
    
    begin_record = False
    sequence = {'figure': figure_number, 'sequence': []}
    for state in game_sequence:
        if state[1]['Time'] >= begin:
            begin_record = True
        if state[1]['Time'] >= end:
            end_record = False
            break
         
        if begin_record:
            sequence['sequence'].append(copy.deepcopy(state))
            # we transform the set() to a list to be able to dump as a json
            sequence['sequence'][-1][0] = list(sequence['sequence'][-1][0])
        
    print('Do you want to add this sequence (y/n): {}'.format(sequence))
    answer = input()
    if answer == 'y':
        sequences.append(copy.deepcopy(sequence))
        with open(PATH_FINAL_OUTPUT, 'w') as f:
            json.dump(sequences, f)
    
    