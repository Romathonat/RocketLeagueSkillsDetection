## How to use
**IMPORTANT NOTE**: This project is not very robust, as I had to make a lot of hacky things to make this work. If psyonix
 decides in the future to include player inputs directly in replays this folder will be outdated.
Note: tested in windows only.
Requires python 3.4 and inputs library (pip install inputs) 

### Get inputs
- Launch get_input_controller.py from powershell. 
- Create a private match on a server, and play alone (simpler). On mutators, select infinite time for match duration, and deactivate "recovery time" (last option)
- When there is the 3-2-1-GO click on the RT or R2 on controller to begin the recording (it used to synchronize with replay).
- Perform skills
- once finished, click on "back" on xbox or "share" on ps4 controller (of course you can also modify the code to change the button to end the recording)
- controller data are saved in current folder as "./input.data"

### Decompile replays 
- look for the replay of the game (usually Documents > MyGames > Rocket League > TAGame > Demos, and decompile it with https://github.com/jjbott/RocketLeagueReplayParser
- This create a .json

### Merge Json and inputs
- launch read_json.py, with the good path (see GLOBAL variables in code)

### Labelize sequences
- Launch the replay viewer.
- Launch extract_sequences.py
- give sequentially the name of the figure, the timing of the beginning, the timing of the end.
- final output is a json file located in the path specified in PATH_FINAL_OUTPUT


