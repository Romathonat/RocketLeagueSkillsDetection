import json
import copy
import math
import pickle
import datetime

#inputs = [[{'up'}, 110], [{'up'}, 112], [{'up'}, 114],[{'down'}, 116]]
# ball starts to move at 10.5

def read_inputs(path, offset, buttons_config, start_replay_button):
    with open(path, 'r') as f:  
        date_begin_inputs = datetime.datetime.strptime(' '.join(f.readline().split()), '%Y-%m-%d %H:%M:%S.%f')
        #inputs = [set([i for i in x.strip().split()]) for x in f.readline().split('-1')]
        inputs = [set([buttons_config[i] if i in buttons_config else i for i in x.strip().split()]) for x in f.readline().split('-1')][:-1]
        
        #print(inputs)
        
        timings = [float(x) + offset for x in f.readline().split()]
        inputs = list(map(list, zip(inputs, timings)))
        
        
        # we remove  command before sync signal    
        for i, input in enumerate(inputs):
            if start_replay_button in input[0]:
                inputs = inputs[i + 1:]
                # we shift to 0 and we add the offset
                zeroing = inputs[0][1]
                inputs = [[copy.deepcopy(i[0]), i[1] - zeroing + offset] for i in inputs] 
                break
                
        # now we remove the accelerate button, which is non-informative. We fusion similar resulting inputs
        inputs_fusioned = []
        previous_input = set()
        
        '''
        for input, timing in inputs:
            input.discard('accelerate')
            if input != previous_input and len(input) > 0:
                inputs_fusioned.append([input, timing])
                previous_input = input
        '''  
    print(inputs[0])
    return inputs

def get_offset_begin_move(frames, players_id, players_pawns, player_name):
    player_id = players_id[player_name]
    actor_position = None
    
    for frame in frames:
        for update in frame['ActorUpdates']:
            if 'ClassName' in update:
                if update['ClassName'] == 'TAGame.Car_TA' and 'Engine.Pawn:PlayerReplicationInfo' in update and update['Engine.Pawn:PlayerReplicationInfo']['ActorId'] == player_id:
                    # we have an update of the pawn of the player
                    players_pawns[player_name] = update['Id']
                
            if update['Id'] == players_pawns[player_name] and 'TAGame.RBActor_TA:ReplicatedRBState' in update:
                # we have an update of the position of the player 
                if actor_position is None:
                    actor_position = copy.deepcopy(update['TAGame.RBActor_TA:ReplicatedRBState']['Position'])
                elif actor_position != update['TAGame.RBActor_TA:ReplicatedRBState']['Position']:
                    # the position has changed
                    print(frame['Time'])
                    return frame['Time']
    return 0
               

def compute_features(last_state, state_time, before_last_state):
    last_state['Time'] = state_time
    try:
        last_state['DistanceBall'] = math.sqrt((last_state['PlayerPosition']['X'] - last_state['BallPosition']['X']) ** 2 +
               (last_state['PlayerPosition']['Y'] - last_state['BallPosition']['Y']) ** 2 +
               (last_state['PlayerPosition']['Z'] - last_state['BallPosition']['Z']) ** 2)
    except KeyError:
        print(last_state)
        last_state['DistanceBall'] = 10000

    try:
        last_state['DistanceWall'] = min(abs(last_state['PlayerPosition']['X'] - WALL_POSITIONS[0]),
                                         abs(last_state['PlayerPosition']['X'] - WALL_POSITIONS[1]))
    except KeyError:
        last_state['DistanceWall'] = 10000

    try:
        last_state['DistanceCeil'] = abs(last_state['PlayerPosition']['Z'] - CEILING_POSITION)

    except KeyError:
        last_state['DistanceCeil'] = 10000

    try:
        last_state['BallSpeed'] = math.sqrt((last_state['BallLinearVelocity']['X']) ** 2 +
               (last_state['BallLinearVelocity']['Y']) ** 2 +
               (last_state['BallLinearVelocity']['Z']) ** 2)
    except (KeyError, TypeError):
        last_state['BallSpeed'] = 0
        
    try:
        #print(before_last_state)
        #print(last_state)
        #print('end')
        last_state['BallAcceleration'] = last_state['BallSpeed'] - before_last_state['BallSpeed']
    except (KeyError, TypeError, ZeroDivisionError) as e:
        last_state['BallAcceleration'] = last_state['BallSpeed']

    try:
        last_state['PlayerSpeed'] = math.sqrt((last_state['PlayerLinearVelocity']['X']) ** 2 +
               (last_state['PlayerLinearVelocity']['Y']) ** 2 +
               (last_state['PlayerLinearVelocity']['Z']) ** 2)
    except (KeyError, TypeError):
        last_state['PlayerSpeed'] = 0
    
    last_state_full = copy.deepcopy(last_state)
    
    if 'BallLinearVelocity' in last_state:
        del last_state['BallLinearVelocity']
    if 'PlayerPosition' in last_state:
        del last_state['PlayerPosition']
    if 'PlayerLinearVelocity' in last_state:
        del last_state['PlayerLinearVelocity']
    if 'BallPosition' in last_state:
        del last_state['BallPosition']

    return last_state_full, last_state

def get_contextual_info(frames, delta_time, last_state, last_timestamp, player_ids, players_pawns, player_name, goals):
    player_id = players_ids[player_name]
    before_last_state = copy.deepcopy(last_state)
    # we reset the goal
    last_state['goal'] = False
    
    
    for frame in frames:
        # we look for states beginning after last_timestamp. We play the succession of states, until we reach the delta_time: we then return the context 
        # note: we take into account only the ball and the player, and we only need their positions and linear velocity
        for update in frame['ActorUpdates']:
            
            if frame['Time'] > delta_time:
                # here we can compute features
                last_state_full, last_state = compute_features(last_state, frame['Time'], before_last_state)
                return last_state_full, last_state, delta_time

            if frame['Time'] < last_timestamp:
                # we do not play before last_timestamp
                continue
            

            # we check if there is a goal between the last timestamp and now, we add it if yes
            for goal_time in goals:
                if goal_time > last_timestamp and goal_time <= frame['Time'] and last_timestamp != 0:
                    # we do not take into account timing 0 else first itemset will be marked as a goal if one has been scored before
                    last_state['goal'] = True

            if 'ClassName' in update:
                if update['ClassName'] == 'TAGame.Car_TA' and 'Engine.Pawn:PlayerReplicationInfo' in update and update['Engine.Pawn:PlayerReplicationInfo']['ActorId'] == player_id:
                    # we have an update of the pawn of the player
                    players_pawns[player_name] = update['Id']
                if update['ClassName'] == 'TAGame.Ball_TA':
                    # we have an update of the ball
                    players_pawns['ball'] = update['Id']

            if update['Id']  == players_pawns[player_name] and 'TAGame.RBActor_TA:ReplicatedRBState' in update:
                # we have an update of the position of the player
                last_state['PlayerLinearVelocity'] = copy.deepcopy(update['TAGame.RBActor_TA:ReplicatedRBState']['LinearVelocity'])
                last_state['PlayerPosition'] = copy.deepcopy(update['TAGame.RBActor_TA:ReplicatedRBState']['Position'])

            if update['Id']  == players_pawns['ball'] and 'TAGame.RBActor_TA:ReplicatedRBState' in update:
                # we have an update of the position of the ball 
                last_state['BallLinearVelocity'] = copy.deepcopy(update['TAGame.RBActor_TA:ReplicatedRBState']['LinearVelocity'])
                last_state['BallPosition'] = copy.deepcopy(update['TAGame.RBActor_TA:ReplicatedRBState']['Position'])

    raise Exception("No data avalaible for this delta_time")


PATH_INPUTS = './replays/replay3/3_inputs.data'
PATH_JSON = './replays/replay3/3.json'
PATH_FINAL_SEQUENCE = './replays/replay3/final3.dat'

PLAYER = 'Erinthril'
BUTTONS_CONFIG = {
    'BTN_WEST': 'slide',
    'BTN_EAST': 'boost',
    'BTN_SOUTH': 'jump',
    'BTN_NORTH': 'camera', 
    'ABS_RZ': 'accelerate', # remove that not useful !
    'ABS_Z': 'slow',
    'LEFT': 'left',
    'RIGHT': 'right',
    'UP': 'up',
    'DOWN': 'down'
}
START_REPLAY_BUTTON = 'accelerate'
WALL_POSITIONS = (-4043, 4043)
CEILING_POSITION = 2030
BACKBOARD_POSITIONS = (-4604, 4604)

with open(PATH_JSON) as json_file:
    data = json.load(json_file)   
    players_ids = {'ball': 0}
    players_pawns = {}
    last_state = {}
    last_state_full = {} # with info on player and ball position
    last_timestamp = 0

    for elt in data['Frames'][0]['ActorUpdates']:
        if 'Engine.PlayerReplicationInfo:PlayerName' in elt:
            players_ids[elt['Engine.PlayerReplicationInfo:PlayerName']] = elt['NameId']
            players_pawns[elt['Engine.PlayerReplicationInfo:PlayerName']] = -1
        if 'ClassName' in elt and elt['ClassName'] == 'TAGame.Ball_TA':
            players_pawns['ball'] = elt['Id']
            last_state['BallPosition'] = elt['InitialPosition']        
    
    # now we get the goals 
    goals = [elt['Time'] for elt in data['TickMarks']]
   
    # now we get the inputs
    inputs = read_inputs(PATH_INPUTS, get_offset_begin_move(data['Frames'], copy.deepcopy(players_ids), copy.deepcopy(players_pawns), PLAYER), BUTTONS_CONFIG, START_REPLAY_BUTTON)
    
    for i, (itemset, time_delta) in enumerate(inputs):  
        last_state_full, last_state, last_timestamp = get_contextual_info(data['Frames'], time_delta, last_state_full, last_timestamp, players_ids, players_pawns, PLAYER, goals)
        inputs[i][1] = copy.deepcopy(last_state)
    print(inputs[15:40])
with open(PATH_FINAL_SEQUENCE, 'wb') as f:
    pickle.dump(inputs, f)
