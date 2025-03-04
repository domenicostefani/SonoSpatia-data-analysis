import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

pickle_file = 'results_and_resampled_data.pickle'

if not os.path.exists('plots'):
    os.makedirs('plots')

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
ALLTRACKS = ['Percussions (ID1)','Xylophone (ID3)', 'Texture (ID5)', 'Brass (ID7)', 'Voice (ID9)']
ALL_AUTOMATIONS = ['Recording Trajectory X / ControlGris', 'Recording Trajectory Y / ControlGris', 'Recording Trajectory Z / ControlGris']

###
# Variability per axis 

for participant in data:
    assert np.all([track in data[participant] for track in ALLTRACKS]), 'Not all tracks found for participant %s'%participant
    for track in ALLTRACKS:
        assert np.all([automation in data[participant][track] for automation in ALL_AUTOMATIONS]), 'Not all automations found for participant %s and track %s'%(participant, track)
        for automation in ALL_AUTOMATIONS:
            assert '2D' in data[participant][track][automation], '2D not found in data (%s)'%data[participant][track][automation].keys()
            assert '3D_DTW' in data[participant][track][automation], '3D_DTW not found in data (%s)'%data[participant][track][automation].keys()
            _2D_mean_stdev = (np.mean(data[participant][track][automation]['2D']), np.std(data[participant][track][automation]['2D']))
            _3D_mean_stdev = (np.mean(data[participant][track][automation]['3D_DTW']), np.std(data[participant][track][automation]['3D_DTW']))
            print(f'Participant {participant} - Track {track} - Automation {automation} - 2D mean/stdev: {_2D_mean_stdev} - 3D mean/stdev: {_3D_mean_stdev}')

PARTICIPANTS = list(data.keys())
# Sort by int id, not string. IDs are of the form ID1, ID2, ..., ID12
PARTICIPANTS.sort(key=lambda x: int(x[2:]))

# Plot means and standard deviations for each participant for all automations
fig, ax = plt.subplots()
opacity = 0.8

for i, participant in enumerate(PARTICIPANTS):
    means_2D = []
    stdevs_2D = []
    means_3D = []
    stdevs_3D = []

    trackAvg2D = [1,1,1]
    trackSD2D = [1,1,1]
    trackAvg3D = [1,1,1]
    trackSD3D = [1,1,1]
    automationlens = [0,0,0]
    for track in ALLTRACKS:
        for aidx, automation in enumerate(ALL_AUTOMATIONS):
            automationlens[aidx] += 1
            trackAvg2D[aidx] += np.mean(data[participant][track][automation]['2D'])
            trackSD2D[aidx] += np.std(data[participant][track][automation]['2D'])
            trackAvg3D[aidx] += np.mean(data[participant][track][automation]['3D_DTW'])
            trackSD3D[aidx] += np.std(data[participant][track][automation]['3D_DTW'])
    for aidx in range(3):
        trackAvg2D[aidx] /= automationlens[aidx]
        trackSD2D[aidx] /= automationlens[aidx]
        trackAvg3D[aidx] /= automationlens[aidx]
        trackSD3D[aidx] /= automationlens[aidx]
        means_2D.append(trackAvg2D[aidx])
        stdevs_2D.append(trackSD2D[aidx])
        means_3D.append(trackAvg3D[aidx])
        stdevs_3D.append(trackSD3D[aidx])


        #     means_2D.append(np.mean(data[participant][track][automation]['2D']))
        #     stdevs_2D.append(np.std(data[participant][track][automation]['2D']))
        #     means_3D.append(np.mean(data[participant][track][automation]['3D']))
        #     stdevs_3D.append(np.std(data[participant][track][automation]['3D']))
    
    x = np.arange(len(means_2D))
    ax.bar(x + i, means_2D, yerr=stdevs_2D, alpha=opacity, label=f'{participant} 2D')
    ax.bar(x + (i+0.1), means_3D, yerr=stdevs_3D, alpha=opacity, label=f'{participant} 3D')

ax.set_xlabel('Automation')
ax.set_ylabel('Mean and standard deviation')
ax.set_title('Mean and standard deviation of MAE per participant')
# ax.set_xticks(x + bar_width)
# ax.set_xticklabels([automation for automation in data[PARTICIPANTS[0]][list(data[PARTICIPANTS[0].keys())[0]]])
ax.legend()

plt.tight_layout()
# plt.savefig('plots/variability_dimensions.png')
plt.show()
