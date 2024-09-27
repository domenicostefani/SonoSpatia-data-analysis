# This time we take the data from "results_and_resampled_data.pickle", perform pca and see the relevance of the pca dimensions


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd


# DATA_3D_TO_USE = '3D_old_noDTW'
DATA_3D_TO_USE = '3D_DTW'







# Load the data
with open("results_and_resampled_data.pickle", "rb") as f:
    data = pickle.load(f)

# Open data/participants.csv
participant_start = {}
with open('data/participants.csv', 'r') as f:
    plines = f.readlines()
assert plines[0].strip().replace(' ','') == 'Participant,Startedwith', 'First line of participants.csv must be "Participant,Start"'

for line in plines[1:]:
    participant, startedwith = tuple([e.strip() for e in line.strip().split(',')])
    assert participant in data.keys(), 'Participant {} not found in data'.format(participant)
    assert startedwith in ['2D', '3D'], 'Startedwith must be either 2D or 3D'
    participant_start[participant] = startedwith
print("Loaded data")


PARALLEL_PCA = True
SERIES_PCA = True

if PARALLEL_PCA:
    for cur_player in data.keys():
        print("\nPlayer", cur_player)
        cur_startedwith = participant_start[cur_player]
        print("\tStarted with", cur_startedwith)

        cur_15D_data2D = pd.DataFrame()
        cur_15D_data3D = pd.DataFrame()

        for cur_track in data[cur_player].keys():
            print("\tTrack", cur_track)
            
            for cur_dimension in data[cur_player][cur_track].keys():
                assert cur_dimension in ['Recording Trajectory X / ControlGris', 'Recording Trajectory Y / ControlGris', 'Recording Trajectory Z / ControlGris'], 'Dimension not recognized'
                
                cur_dimensionstr = cur_dimension.replace('Recording Trajectory ', '').replace(' / ControlGris', '').strip()
                assert cur_dimensionstr in ['X', 'Y', 'Z'], 'Dimension not recognized'

                cur_track_str = cur_track.replace('Track','').split(' ')[0].strip()
                assert cur_track_str in ['Percussions','Xylophone','Texture','Brass','Voice']

                cur_colname = cur_track_str + '_' + cur_dimensionstr

                cur_data = data[cur_player][cur_track][cur_dimension][DATA_3D_TO_USE]

                cur_15D_data3D[cur_colname] = cur_data

                cur_data = data[cur_player][cur_track][cur_dimension]['2D']
                cur_15D_data2D[cur_colname] = cur_data

    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
        
        max_explained_variance = 0.0
        for cur_15D_data,type2D3D, ax in zip([cur_15D_data2D, cur_15D_data3D],['2D','3D'], (ax1, ax2)):

            # print(cur_15D_data.head())
            # assert cur_15D_data.shape[1] == 15, 'Data shape is not 15'
            assert cur_15D_data.shape[1] <= 15, 'Data shape is > 15'
            dimensions = cur_15D_data.shape[1]
            # Perform PCA
            print('dimensions', dimensions)
            pca = PCA(n_components=dimensions)
            pca.fit(cur_15D_data)
            print("Explained variance ratio", pca.explained_variance_ratio_)
            # Plot as bars
            if type2D3D == '3D':
                ax.bar(range(1, dimensions+1), pca.explained_variance_ratio_, color='orange')
            else:
                ax.bar(range(1, dimensions+1), pca.explained_variance_ratio_)
            ax.set_xlabel('Principal component')
            ax.set_ylabel('Explained variance ratio')
            titll = 'PCA explained variance\np' + cur_player + ' '+type2D3D+' started with ' + cur_startedwith
            ax.set_title(titll)
            max_explained_variance = max(max_explained_variance, np.max(pca.explained_variance_ratio_))


            print("Mean", pca.mean_)
            print("Noise variance", pca.noise_variance_)
            print("n_components", pca.n_components_)

        ax1.set_ylim([0, max_explained_variance*1.15])
        ax2.set_ylim([0, max_explained_variance*1.15])

        import os
        if not os.path.exists('plots/pcas'):
            os.makedirs('plots/pcas')
        plt.savefig('plots/pcas/PCA_explained_variance_'+cur_player+'.png')
        # os.system('')
        # plt.show()
