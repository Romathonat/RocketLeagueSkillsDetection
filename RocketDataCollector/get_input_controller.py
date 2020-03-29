from datetime import datetime
from inputs import get_gamepad

sequence = []
sequence_timing = []
current_events = set()
time_begin = datetime.now()
JOYSTICK_THRESHOLD = 10000
start_recording = False

def add_to_sequence(current_events, sequence, sequence_timing):
    if len(current_events) != 0:
        sequence.append(current_events.copy())
        sequence_timing.append((datetime.now() - time_begin).total_seconds())
        

def format_output(sequence):
    output = ''
    for itemset in sequence:
        output_itemset = ''
        for item in itemset:
            output_itemset += '{} '.format(item)
        output_itemset += '-1 '
        output += output_itemset
    output += '-2'
    return output

while 1:
    events = get_gamepad()
    for event in events:
        if event.code not in ('SYN_REPORT', 'ABS_RX', 'ABS_RY'):
            #we ignore the right joystick which is for camera
            if event.code == 'ABS_RZ' and not start_recording:
                print('Start recording')
                start_recording = True
            if not start_recording :
                continue
            
            if event.code == 'BTN_START':
                with open('./inputs.data', 'w') as f:
                    f.write(str(time_begin))    
                    f.write('\n')
                    f.write(format_output(sequence))
                    f.write('\n')
                    f.write(' '.join([str(x) for x in sequence_timing]))
                exit()
            elif event.code == 'ABS_X':
                # deal with stick x
                if event.state > JOYSTICK_THRESHOLD and 'RIGHT' not in current_events:
                    current_events.add('RIGHT')
                    add_to_sequence(current_events, sequence, sequence_timing)
                elif event.state < -JOYSTICK_THRESHOLD and 'LEFT' not in current_events:
                    current_events.add('LEFT')
                    add_to_sequence(current_events, sequence, sequence_timing)
                elif event.state < JOYSTICK_THRESHOLD and event.state >= 0 and 'RIGHT' in current_events:
                    current_events.remove('RIGHT')
                    add_to_sequence(current_events, sequence, sequence_timing)
                elif event.state > -JOYSTICK_THRESHOLD and event.state <= 0 and 'LEFT' in current_events:
                    current_events.remove('LEFT')
                    add_to_sequence(current_events, sequence, sequence_timing)
            elif event.code == 'ABS_Y':
                # deal with stick y
                if event.state > JOYSTICK_THRESHOLD and 'UP' not in current_events:
                    current_events.add('UP')
                    add_to_sequence(current_events, sequence, sequence_timing)
                elif event.state < -JOYSTICK_THRESHOLD and 'DOWN' not in current_events:
                    current_events.add('DOWN')
                    add_to_sequence(current_events, sequence, sequence_timing)
                elif event.state < JOYSTICK_THRESHOLD and event.state >= 0 and 'UP' in current_events:
                    current_events.remove('UP')
                elif event.state > -JOYSTICK_THRESHOLD and event.state <= 0 and 'DOWN' in current_events:
                    current_events.remove('DOWN')
            elif event.state != 0 and event.code not in current_events:          
                # is pressed
                current_events.add(event.code)
                add_to_sequence(current_events, sequence, sequence_timing)
            elif event.state == 0:      
                current_events.remove(event.code)
                add_to_sequence(current_events, sequence, sequence_timing)               
                
            