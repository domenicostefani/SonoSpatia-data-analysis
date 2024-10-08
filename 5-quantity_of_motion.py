import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

pickle_file = 'results_and_resampled_data.pickle'

if not os.path.exists('plots'):
    os.makedirs('plots')

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Open data/participants.csv
participant_start = {}
with open('data/participants.csv', 'r') as f:
    plines = f.readlines()
assert plines[0].strip().replace(' ','') == 'Participant,Startedwith', 'First line of participants.csv must be "Participant,Start"'

for line in plines[1:]:
    participant, startedwith = line.strip().split(',')
    participant = participant.strip()
    assert participant in data.keys(), 'Participant {} not found in data'.format(participant)

    startedwith = startedwith.strip()
    assert startedwith in ['2D', '3D'], 'Startedwith must be either 2D or 3D'
    participant_start[participant] = startedwith





data.keys()
