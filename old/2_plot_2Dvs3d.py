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


# Create barplot of MAE with a bar for each participant and for automation 0
bar_width = 0.35
part_spacing = 0.5
opacity = 0.8

maes_X = []
maes_X_startedwith2D = []
maes_X_startedwith3D = []
maes_Y = []
maes_Y_startedwith2D = []
maes_Y_startedwith3D = []
maes_Z = []
maes_Z_startedwith2D = []
maes_Z_startedwith3D = []

for participant in data:
    alltrack_maesX = []
    alltrack_maesY = []
    alltrack_maesZ = []
    for track in data[participant]:
        automationX = 'Recording Trajectory X / ControlGris'
        automationY = 'Recording Trajectory Y / ControlGris'
        automationZ = 'Recording Trajectory Z / ControlGris'
        if automationX in data[participant][track]:
            alltrack_maesX.append(data[participant][track][automationX]['metrics']['mae'])

        if automationY in data[participant][track]:
            alltrack_maesY.append(data[participant][track][automationY]['metrics']['mae'])

        if automationZ in data[participant][track]:
            alltrack_maesZ.append(data[participant][track][automationZ]['metrics']['mae'])

    
    maes_X_towrite = np.mean(alltrack_maesX) if len(alltrack_maesX) > 0 else 1.0
    maes_X.append(maes_X_towrite)

    maes_Y_towrite = np.mean(alltrack_maesY) if len(alltrack_maesY) > 0 else 1.0
    maes_Y.append(maes_Y_towrite)

    maes_Z_towrite = np.mean(alltrack_maesZ) if len(alltrack_maesZ) > 0 else 1.0
    maes_Z.append(maes_Z_towrite)

    if participant_start[participant] == '2D':
        maes_X_startedwith2D.append(maes_X_towrite)
        maes_Y_startedwith2D.append(maes_Y_towrite)
        maes_Z_startedwith2D.append(maes_Z_towrite)
    elif participant_start[participant] == '3D':
        maes_X_startedwith3D.append(maes_X_towrite)
        maes_Y_startedwith3D.append(maes_Y_towrite)
        maes_Z_startedwith3D.append(maes_Z_towrite)
    else:
        raise ValueError('Participant {} has an invalid startedwith value'.format(participant))
    
maes_startedwith2D = (maes_X_startedwith2D, maes_Y_startedwith2D, maes_Z_startedwith2D)
maes_startedwith3D = (maes_X_startedwith3D, maes_Y_startedwith3D, maes_Z_startedwith3D)

assert len(maes_X) == len(maes_Y) == len(maes_Z), 'X, Y, and Z must have the same length'
assert len(maes_X) == len(list(data.keys())), 'X, Y, Z, and data must have the same length'
            
        # maes.append(np.mean(alltrack_maes))


# matplotlib default colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


DO_PLOT_PER_PARTICIPANT = False

if DO_PLOT_PER_PARTICIPANT:
    fig, ax = plt.subplots()

    # print(maes)
    enlargeX = 1.2
    x_points = np.arange(len(maes_X)) * enlargeX
    for i,posx in enumerate(x_points):
        ax.bar(posx,               maes_X[i], bar_width, alpha=opacity, color = colors[0], label='X' if i == 0 else '')
        ax.bar(posx + bar_width,   maes_Y[i], bar_width, alpha=opacity, color = colors[1], label='Y' if i == 0 else '')
        ax.bar(posx + 2*bar_width, maes_Z[i], bar_width, alpha=opacity, color = colors[2], label='Z' if i == 0 else '')


    plt.xlabel('Participant')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error for each participant')
    plt.xticks(x_points+ bar_width, np.arange(1, len(maes_X)+1))
    plt.legend()

    plt.savefig('plots/ALLmae.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/ALLmae.pdf', dpi=300, bbox_inches='tight')
    plt.show()






## Now barplot collapsing all dimensions into one bar per participant
allmaes = []
stderror = []
for i in range(len(maes_X)):
    allmaes.append((maes_X[i]+maes_Y[i]+maes_Z[i])/3.0)
    stderror.append(np.std([maes_X[i],maes_Y[i],maes_Z[i]])/np.sqrt(3.0))


DO_PLOT_PER_PARTICIPANT2 = False

if DO_PLOT_PER_PARTICIPANT2:
    fig, ax = plt.subplots()
    enlargeX = 1.2
    x_points = np.arange(len(allmaes)) * enlargeX
    for i,posx in enumerate(x_points):
        ax.bar(posx, allmaes[i], alpha=opacity, color = colors[0], yerr=stderror[i], capsize=5)

    plt.xlabel('Participant')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error between 2D and 3D')
    plt.xticks(x_points, np.arange(1, len(allmaes)+1))


    plt.savefig('plots/mae_per_participant.png', dpi=300, bbox_inches='tight')
    plt.savefig('plots/mae_per_participant.pdf', dpi=300, bbox_inches='tight')
    plt.show()



# Now barplot collapsing all participants into one bar per dimension
allmaes = []
stderror = []

for mmae in [maes_X, maes_Y, maes_Z]:
    allmaes.append(np.mean(mmae))
    stderror.append(np.std(mmae)/np.sqrt(len(mmae)))

fig, ax = plt.subplots()
enlargeX = 1.2
x_points = np.arange(len(allmaes)) * enlargeX
for i,posx in enumerate(x_points):
    ax.bar(posx, allmaes[i], alpha=opacity, color = colors[1], yerr=stderror[i], capsize=5)

xtickspos = []
for posx, dimension in zip(x_points,['X', 'Y', 'Z']):
    xtickspos.append((posx, dimension))
plt.xticks([xt[0] for xt in xtickspos], [xt[1] for xt in xtickspos])

plt.xlabel('Dimension')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error between 2D and 3D - per dimension')


plt.savefig('plots/mae_per_dimension.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/mae_per_dimension.pdf', dpi=300, bbox_inches='tight')
plt.show()



# Now we split between the participants that started with 2D and 3D
toplot_byFirst = {'2D':[], '3D':[]}
toplot_err_byFirst = {'2D':[], '3D':[]}
for startedWith,curmaes in zip(['2D','3D'],[maes_startedwith2D,maes_startedwith3D]):
    # print('Started with {}'.format(startedWith))

    allmaes = []
    stderror = []

    for mmae in curmaes:
        allmaes.append(np.mean(mmae))
        toplot_byFirst[startedWith].append(np.mean(mmae))
        stderror.append(np.std(mmae)/np.sqrt(len(mmae)))
        toplot_err_byFirst[startedWith].append(np.std(mmae)/np.sqrt(len(mmae)))
    assert len(allmaes) == 3, 'There should be 3 dimensions'

    fig, ax = plt.subplots()
    enlargeX = 1.2
    x_points = np.arange(len(allmaes)) * enlargeX
    for i,posx in enumerate(x_points):

        ax.bar(posx, allmaes[i], alpha=opacity, color = colors[1], yerr=stderror[i], capsize=5)

    xtickspos = []
    for posx, dimension in zip(x_points,['X', 'Y', 'Z']):
        xtickspos.append((posx, dimension))
    plt.xticks([xt[0] for xt in xtickspos], [xt[1] for xt in xtickspos])

    plt.xlabel('Dimension')
    plt.ylabel('Mean Absolute Error')
    plt.title('Mean Absolute Error between 2D and 3D,\n for participants who started with {}'.format(startedWith))


    plt.savefig('plots/mae_per_dimension_started_with_%s.png'%startedWith, dpi=300, bbox_inches='tight')
    plt.savefig('plots/mae_per_dimension_started_with_%s.pdf'%startedWith, dpi=300, bbox_inches='tight')
    plt.show()

fig, ax = plt.subplots()


ind = np.arange(len(toplot_byFirst['2D']))    # the x locations for the groups

# Width of a bar 
width = 0.3       

# Plotting
plt.bar(ind, toplot_byFirst['2D'] , width, label='2D-first', yerr=toplot_err_byFirst['2D'], capsize=3)
plt.bar(ind + width, toplot_byFirst['3D'], width, label='3D-first', yerr=toplot_err_byFirst['3D'], capsize=3)

# draw vertocal lines at ind
# for x in ind:
#     plt.axvline(x=x+width/2, color='black', linestyle='--', lw=1)

# Put xticks at the middle of the bars
plt.xticks(ind + width/2, ['X', 'Y', 'Z'])

# plt.ylim(0, 1)

plt.legend()
plt.xlabel('Dimension')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error between 2D and 3D')


# fig, ax = plt.subplots()
# enlargeX = 1.2
# x_points = np.arange(len(allmaes)) * enlargeX

# ax.boxplot([maes_X, maes_Y, maes_Z], positions=x_points, widths=0.6, patch_artist=True, showmeans=True, meanline=True, meanprops=dict(color='black', linewidth=2))

# xtickspos = []
# for posx, dimension in zip(x_points,['X', 'Y', 'Z']):
#     xtickspos.append((posx, dimension))
# plt.xticks([xt[0] for xt in xtickspos], [xt[1] for xt in xtickspos])

# plt.xlabel('Dimension')
# plt.ylabel('Mean Absolute Error')
# plt.title('Mean Absolute Error between 2D and 3D - per dimension')
    
plt.savefig('plots/mae_per_dimension_bystart.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/mae_per_dimension_bystart.pdf', dpi=300, bbox_inches='tight')
plt.show()




