import pickle
from utils import get_point_list_from_automation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

EXPERIMENT_TYPES = ['2D', '3D_DTW','3D_old_noDTW']
ENDFILE_SECONDS = 29.538 # End of task in seconds, throw away subsequent points
MAX_ALLOWED_DTWSHIFT_SECONDS = 3.0 # Maximum allowed shift in seconds by DTW
VERBOSE_MAXSHIFT = False
DO_PLOT_DTW_PARTIAL = True
DO_PLOT_DTW_PARTIAL_JUST_FIRST = False

with open('extracted_envelopes.pickle', 'rb') as f:
    per_participant_automation_dict = pickle.load(f)

# print(per_participant_automation_dict.keys())
# dict_keys(['ID10', 'ID11', 'ID12', 'ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8', 'ID9'])

# print(per_participant_automation_dict['ID1'].keys())
# ['2D', '3D']

# print(per_participant_automation_dict['ID1']['2D'].keys())
# ['Percussions (ID1)', 'Xylophone (ID3)', 'Texture (ID5)', 'Brass (ID7)', 'Voice (ID9)']

# print(len(per_participant_automation_dict['ID1']['2D']['Percussions (ID1)']))
# 3

# print(per_participant_automation_dict['ID1']['2D']['Percussions (ID1)'][0].keys())
# ['index', 'name', 'track', 'automation']

# print(type(per_participant_automation_dict['ID1']['2D']['Percussions (ID1)'][0]['automation']))
# list

PARTICIPANTS = list(per_participant_automation_dict.keys())
ALLTRACKS = ['Percussions (ID1)','Xylophone (ID3)', 'Texture (ID5)', 'Brass (ID7)', 'Voice (ID9)']


# Sort by int id, not string. IDs are of the form ID1, ID2, ..., ID12
PARTICIPANTS.sort(key=lambda x: int(x[2:]))

results = {}
# 
max_dtw_shift = 0
max_dtw_shift_info = None
for partidx,participant in enumerate(PARTICIPANTS):
    print('Processing participant',participant)
    cur_automation_dict_2D = per_participant_automation_dict[participant]['2D']
    cur_automation_dict_3D = per_participant_automation_dict[participant]['3D']

    cur_results = results[participant] = {}

    assert all([track in cur_automation_dict_2D.keys() for track in ALLTRACKS]), 'Participant {} is missing some tracks in 2D dict'.format(participant)
    assert all([track in cur_automation_dict_3D.keys() for track in ALLTRACKS]), 'Participant {} is missing some tracks in 3D dict'.format(participant)
    # all_tracks = set(cur_automation_dict_2D.keys()).union(set(cur_automation_dict_3D.keys()))
    # assert len(all_tracks) == 5, 'Expected 5 tracks, found {}'.format(len(all_tracks))
    all_tracks = ALLTRACKS

    for tidx,track in enumerate(all_tracks): #cur_automation_dict_2D:

        cur_results_track = cur_results[track] = {}

        print('  Processing track',track)
        assert track in cur_automation_dict_3D, 'Track not found in 3D automation'
        assert track in cur_automation_dict_2D, 'Track not found in 2D automation'

        all_dimension_automations = set([e['name'] for e in cur_automation_dict_2D[track]]).union(set([e['name'] for e in cur_automation_dict_3D[track]]))

        assert len(all_dimension_automations) == 3, 'Expected 3 dimensions, found {}'.format(len(all_dimension_automations))
        assert len(cur_automation_dict_2D[track]) == 3, 'Expected 3 dimensions in 2D automations, found {}'.format(len(cur_automation_dict_2D[track]))
        assert len(cur_automation_dict_3D[track]) == 3, 'Expected 3 dimensions in 3D automations, found {}'.format(len(cur_automation_dict_3D[track]))
        for automation_idx in range(3):
            # TODO: THERE IS A BUG HERE (But not for our case)
            # If you skip automation X or Y, index will be different
            assert cur_automation_dict_2D[track][automation_idx]['name'] == cur_automation_dict_3D[track][automation_idx]['name'], 'Automation name mismatch, fix code to cycle through names instead of indexes'
            name = cur_automation_dict_2D[track][automation_idx]['name']

            cur_results_automation = cur_results_track[name] = {}

            print('    Processing automation',cur_automation_dict_2D[track][automation_idx]['name'], 'index:',automation_idx)

            # print('      len(2D):',len(cur_automation_dict_2D[track][automation_idx]['automation']))
            # print('      len(3D):',len(cur_automation_dict_3D[track][automation_idx]['automation']))
           
            T_toplot_2D, V_toplot_2D = get_point_list_from_automation(cur_automation_dict_2D[track][automation_idx])
            T_toplot_3D, V_toplot_3D = get_point_list_from_automation(cur_automation_dict_3D[track][automation_idx])

            # assert np.less_equal(T_toplot_2D, ENDFILE_SECONDS).all(), '2D Envelope {} for track {}, participant {} contains time values greater than {:f} ({:f}>{:f})'.format(name, track, participant,ENDFILE_SECONDS,max(T_toplot_2D),ENDFILE_SECONDS)
            # assert np.less_equal(T_toplot_3D, ENDFILE_SECONDS).all(), '3D Envelope {} for track {}, participant {} contains time values greater than {:f} ({:f}>{:f})'.format(name, track, participant,ENDFILE_SECONDS,max(T_toplot_3D),ENDFILE_SECONDS)

            assert not np.any(np.isnan(V_toplot_2D)), 'V_toplot_2D contains NaNs'
            assert not np.any(np.isnan(V_toplot_3D)), 'V_toplot_3D contains NaNs'
            assert not np.any(np.isnan(T_toplot_2D)), 'T_toplot_2D contains NaNs'
            assert not np.any(np.isnan(T_toplot_3D)), 'T_toplot_3D contains NaNs'

            # print('2D:',len(T_toplot_2D),'points')
            # print('3D:',len(T_toplot_3D),'points')


            # DO_PLOT = True
            # if DO_PLOT:
            #     plt.figure(figsize=(18, 3))
            #     plt.title('RAW 2D vs 3D automation\n{}, {}, {}'.format(participant, track, cur_automation_dict_2D[track][automation_idx]['name']))
            #     # Plot lines with round markers
            #     plt.plot(T_toplot_2D, V_toplot_2D, 'o-', label='2D', color='red')
            #     plt.plot(T_toplot_3D, V_toplot_3D, 'o-', label='3D', color='blue')
            #     plt.legend()

            #     plt.ylim(0,1)
            #     plt.xlim(0,30)

            from scipy.interpolate import interp1d

            # Before interpolating we solve issues with 1-element arrays.
            # We do so by adding a point at the end of the array with the same value as the last point and at time 30ms (end of task)

            if T_toplot_2D[-1] < ENDFILE_SECONDS:
                T_toplot_2D.append(ENDFILE_SECONDS)
                V_toplot_2D.append(V_toplot_2D[-1])

            if T_toplot_3D[-1] < ENDFILE_SECONDS:
                T_toplot_3D.append(ENDFILE_SECONDS)
                V_toplot_3D.append(V_toplot_3D[-1])

            interp_func_2D = interp1d(np.array(T_toplot_2D), np.array(V_toplot_2D), kind='linear', fill_value="extrapolate")
            interp_func_3D = interp1d(np.array(T_toplot_3D), np.array(V_toplot_3D), kind='linear', fill_value="extrapolate")
            # interp_func_2D = interp1d(x1, y1, kind='linear', fill_value=(0.0,1.0), bounds_error=False)
            # interp_func_3D = interp1d(x2, y2, kind='linear', fill_value=(0.0,1.0), bounds_error=False)


            # PLOT_BEFORE_AFTER = True
            # if PLOT_BEFORE_AFTER == True:
            #     plt.figure(figsize=(18, 3))
            #     plt.title('Before/After 2D interpolation')
            #     # Plot lines with round markers
            #     plt.plot(T_toplot_2D, V_toplot_2D, 'o-', label='2D original', color='red')
            #     plt.plot(T_toplot_2D, interp_func_2D(T_toplot_2D), 'o-', label='2D interpolated', color='green')

            #     plt.legend()
            #     # plt.show()

            #     plt.figure(figsize=(18, 3))
            #     plt.title('Before/After 3D interpolation')
            #     # Plot lines with round markers
            #     plt.plot(T_toplot_3D, V_toplot_3D, 'o-', label='3D original', color='blue')
            #     plt.plot(T_toplot_3D, interp_func_3D(T_toplot_3D), 'o-', label='3D interpolated', color='green')

            #     plt.legend()

            assert max(min(T_toplot_2D), min(T_toplot_3D)) < 0.1, 'Envelope {} for track {}, participant {} does not start at 0 but at {}'.format(name, track, participant, max(min(T_toplot_2D), min(T_toplot_3D)))

            t_min = max(min(T_toplot_2D), min(T_toplot_3D))
            assert t_min == 0.0
            t_max = min(ENDFILE_SECONDS,max(T_toplot_2D), max(T_toplot_3D))

            assert t_max == ENDFILE_SECONDS, 'Envelope {} for track {}, participant {} does not end at 30 but at {}'.format(name, track, participant, t_max)

            # print('Creating a common x-axis from {} to {}'.format(t_min, t_max))
            common_t = np.linspace(t_min, t_max, num=1000)  # 5000 points for high resolution
            

            # PLOT_BEFORE_AFTER = True
            # if PLOT_BEFORE_AFTER == True:
            #     plt.figure(figsize=(18, 3))
            #     plt.title('Before/After 2D interpolation AND RESAMPLING')
            #     # Plot lines with round markers
            #     plt.plot(T_toplot_2D, V_toplot_2D, 'o-', label='2D original', color='red')
            #     plt.plot(common_t, interp_func_2D(common_t), 'o-', label='2D interpolated', color='green')

            #     plt.legend()
            #     # plt.show()

            #     plt.figure(figsize=(18, 3))
            #     plt.title('Before/After 3D interpolation AND RESAMPLING')
            #     # Plot lines with round markers
            #     plt.plot(T_toplot_3D, V_toplot_3D, 'o-', label='3D original', color='blue')
            #     plt.plot(common_t, interp_func_3D(common_t), 'o-', label='3D interpolated', color='green')

            #     plt.legend()


            # # Evaluate both interpolated functions on the common x-values
            v2D_resampled = list(interp_func_2D(common_t))
            v3D_resampled = list(interp_func_3D(common_t))

            cur_results_automation['2D'] = v2D_resampled
            # cur_results_automation['3D'] = v3D_resampled # We will only store the resampled 3D after DTW


            # DO_PLOT = True
            # if DO_PLOT:
            #     plt.figure(figsize=(18, 3))
            #     plt.title('Resampled 2D vs 3D automation')
            #     # Plot lines with round markers
            #     plt.plot(common_t, v2D_resampled, 'o-', label='2D', color='red')
            #     plt.plot(common_t, v3D_resampled, 'o-', label='3D', color='blue')
            #     plt.legend()

            #     plt.ylim(0,1)
            #     plt.xlim(0,30)
            #     plt.show()


            def plot_raw_automation(ts, vs, title):
                plt.figure(figsize=(18, 3))
                plt.title(title)
                plt.plot(ts, vs, 'o-', color='blue')
                plt.ylim(0,1)
                plt.xlim(0,30)


            ######
            # Look for NaNs
            if np.any(np.isnan(v2D_resampled)):
                print('    WARNING: 2D automation {} contains NaNs'.format(name))
                print('    2D Automation had {} points'.format(len(T_toplot_2D)))
                # Print indexes
                print('    NaN indexes:',np.where(np.isnan(v2D_resampled)))

                for idx in range(10):
                    print('(%.2f, %.2f)'%(T_toplot_2D[idx], V_toplot_2D[idx]))


                plot_raw_automation(T_toplot_2D, V_toplot_2D, '2D automation')
                plt.show()
                exit(1)

            if np.any(np.isnan(v3D_resampled)):
                print('    WARNING: 3D automation {} contains NaNs'.format(name))
                print('    3D Automation had {} points'.format(len(T_toplot_3D)))
                # Print indexes
                print('    NaN indexes:',np.where(np.isnan(v3D_resampled)))
                for idx in range(10):
                    print('(%.2f, %.2f)'%(T_toplot_3D[idx], V_toplot_3D[idx]))


                plot_raw_automation(T_toplot_3D, V_toplot_3D, '3D automation')
                plt.show()
                exit(1)



            query = np.array(v2D_resampled)
            template = np.array(v3D_resampled)

            from dtw import *

            is_first = partidx == 0 and tidx == 0 and automation_idx == 0

            # if DO_PLOT_DTW_PARTIAL:
            #     if (DO_PLOT_DTW_PARTIAL_JUST_FIRST and is_first) or not DO_PLOT_DTW_PARTIAL_JUST_FIRST:
            #         ## Display the warping curve, i.e. the alignment curve
            #         alignment.plot(type="threeway")

            ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
            assert len(query) == len(template), 'Query and template must have the same length'
            assert len(query) == 1000, 'Query and template must have 1000 points'
            # dtw_alignment = dtw(query, template, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
            # dtw_alignment = dtw(query, template, keep_internals=True, window_type="sakoechiba", window_args={'window_size': 0.001})

            index_to_time = lambda idx: common_t[int(idx)]
            time_to_index = lambda time: np.argmin(np.abs(common_t - time))
            

            max_allowed_dtwdhift_inindexes = time_to_index(MAX_ALLOWED_DTWSHIFT_SECONDS)

            # dtw_alignment = dtw(query, template, keep_internals=True, window_type="sakoechiba", window_args={'window_size': float(max_allowed_dtwdhift_inindexes)})
            dtw_alignment = dtw(query,
                                template,
                                keep_internals=True,
                                step_pattern=rabinerJuangStepPattern(6, "c"),
                                window_type="sakoechiba",
                                window_args={'window_size': float(max_allowed_dtwdhift_inindexes)})
                                # window_args={'window_size': 10})
            

            if DO_PLOT_DTW_PARTIAL:
                if (DO_PLOT_DTW_PARTIAL_JUST_FIRST and is_first) or not DO_PLOT_DTW_PARTIAL_JUST_FIRST:
                    
                    OFFSET = -2
                    dtw_alignment.plot(type="twoway",offset=OFFSET)

                    # print('maxtime',max(common_t))
                    # reconvert 0-1000 index to time (0-30s ~)
                    
                    # print('index_to_time(0)',index_to_time(0))

                    # Desired ticks show labels every X seconds and one for last mark which is float
                    desired_ticks = np.arange(0, max(common_t), 5)
                    if desired_ticks[-1] < max(common_t):
                        desired_ticks = np.append(desired_ticks, max(common_t))
                    # print('desired_ticks (%d):'%len(desired_ticks),desired_ticks)
                    xticks = [time_to_index(e) for e in desired_ticks]
                    # print('xticks (%d):'%len(xticks),xticks)

                    xticklabels = [str(e) for e in desired_ticks]
                    plt.xticks(xticks, xticklabels)
                    
                    ### Compute here the maximum shift applied by DTW
                    cur_maxshift_raw = max(abs(dtw_alignment.index1 - dtw_alignment.index2))
                    cur_maxshift_pos = -1
                    cur_maxshift_pos2 = -1
                    for i in range(len(dtw_alignment.index1)):
                        if abs(dtw_alignment.index1[i] - dtw_alignment.index2[i]) == cur_maxshift_raw:
                            cur_maxshift_pos = dtw_alignment.index1[i]
                            cur_maxshift_pos2 = dtw_alignment.index2[i]
                            break

                    # vertical line at max shift position
                    plt.axvline(x=cur_maxshift_pos, color='gray', alpha=0.5)
                    plt.axvline(x=cur_maxshift_pos2, color='gray', alpha=0.5)

                    point1 = (cur_maxshift_pos, query[cur_maxshift_pos]-OFFSET)
                    point2 = (cur_maxshift_pos2, template[cur_maxshift_pos2])
                    # plt.axhline(y=point1[1], color='r', linestyle='--')
                    # plt.axhline(y=point2[1], color='r', linestyle='--')
                    
                    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r--')
                    # Add also circles
                    plt.plot(point1[0], point1[1], 'ro')
                    plt.plot(point2[0], point2[1], 'ro')
                    
                    assert cur_maxshift_pos != -1, 'Max shift position not found'

                    print('cur_maxshift_raw: ',cur_maxshift_raw)
                    print('cur_maxshift_pos: ',cur_maxshift_pos)
                    

                    # ensure index1 is always <= len(common_t) and index2 is always <= len(common_t)
                    assert max(dtw_alignment.index1) <= len(common_t), 'Max index1 is greater than length of common_t, (%f > %f)'%(max(dtw_alignment.index1), len(common_t))
                    assert max(dtw_alignment.index2) <= len(common_t), 'Max index2 is greater than length of common_t, (%f > %f)'%(max(dtw_alignment.index2), len(common_t))
                    assert min(dtw_alignment.index1) >= 0, 'Min index1 is less than 0, (%f < 0)'%(min(dtw_alignment.index1))
                    assert min(dtw_alignment.index2) >= 0, 'Min index2 is less than 0, (%f < 0)'%(min(dtw_alignment.index2))


                    assert cur_maxshift_raw <= len(common_t), 'Max shift index is greater than length of common_t, (%f > %f)'%(cur_maxshift_raw, len(common_t))
                    assert cur_maxshift_pos <= len(common_t), 'Max shift position index is greater than length of common_t (%f > %f)'%(cur_maxshift_pos, len(common_t))
                    print('Max shift:',cur_maxshift_raw)
                    cur_maxshift_s = float(cur_maxshift_raw * (ENDFILE_SECONDS / len(common_t)))
                    if cur_maxshift_s > max_dtw_shift:
                        max_dtw_shift = cur_maxshift_s
                        max_dtw_shift_info =  {'participant':participant,
                                               'track':track,
                                               'automation':name,
                                               'max_shift_seconds':cur_maxshift_s,
                                               'max_shift_position_seconds':float(index_to_time(cur_maxshift_pos)),
                                               'max_shift_index':int(cur_maxshift_raw),
                                               'max_shift_position_index':int(cur_maxshift_pos)}
                    if VERBOSE_MAXSHIFT:
                        print('Max shift in seconds: %.2f'%cur_maxshift_s)
                        print('max_shift_position_seconds: %.2f'%float(index_to_time(cur_maxshift_pos)))
                    plt.title('DTW alignment,\nplayer %s, track %s,\nautomation %s\n MaxShift:%.1f s at mark %.1fs'%(participant, track, name,cur_maxshift_s, float(index_to_time(cur_maxshift_pos))))

                    # exit()

                    plotdir = 'plots/dtw/'
                    if is_first and not os.path.exists(plotdir):
                        os.makedirs(plotdir)
                    plotfname = ('dtw_alignment_%s_%s_%s.jpg'%(participant, track, name)).replace(' ','_').replace('/','_').replace('(','_').replace(')','_')
                    plotfname = os.path.join(plotdir, plotfname)
                    plt.savefig(plotfname, bbox_inches='tight')
                    # plt.show()
                    plt.close()
                    # exit()


            # After DTW we resample the 3D automation
            # But we make sure to reconvert 0-1000 index to time (0-30s ~)
            index_to_time = lambda idx: common_t[int(idx)]
            interp_func_PostDTW_3D = interp1d([index_to_time(e) for e in dtw_alignment.index1], template[dtw_alignment.index2], kind='linear', fill_value="extrapolate")


            # plt.plot(query, label='2D') 
            # plt.plot(template, label='3D-raw')                                     
            # plt.plot(dtw_alignment.index1,template[dtw_alignment.index2], label='3D aligned')
            # plt.legend()

            # plt.show()


            v3D_DTW_resampled = list(interp_func_PostDTW_3D(common_t))

            DO_PLOT_DTW_RESAMPLING = False
            if DO_PLOT_DTW_RESAMPLING:
                # plt.plot(query, label='2D') 
                # plt.plot(template, label='3D-raw')          
                fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(18, 8))                           
                ax1.plot(common_t,template, label='3D raw', color='b')


                ax2.plot([index_to_time(e) for e in dtw_alignment.index1],template[dtw_alignment.index2], label='3D DTW', color='r')
                ax3.plot(common_t, v3D_DTW_resampled, label='3D DTW resampled', color='g', linestyle='dashed')   

                fig.legend()
                plt.show()

            
            cur_results_automation['3D_old_noDTW'] = v3D_resampled
            del v3D_resampled
            cur_results_automation['3D_DTW'] = v3D_DTW_resampled

            
            cur_results_automation['metrics'] = {}
            cur_results_automation['metrics']['description'] = 'Metrics computed:\n'

            mae = float(mean_absolute_error(v2D_resampled, v3D_DTW_resampled))
            cur_results_automation['metrics']['mae'] = mae
            cur_results_automation['metrics']['description'] += 'Mean Absolute Error of 2D automation vs 3D: %.2f\n'%mae

            mse = float(mean_squared_error(v2D_resampled, v3D_DTW_resampled))
            cur_results_automation['metrics']['mse'] = mse
            cur_results_automation['metrics']['description'] += 'Mean Squared Error of 2D automation vs 3D: %.2f\n'%mse
            rmse = np.sqrt(mse)
            cur_results_automation['metrics']['rmse'] = rmse
            cur_results_automation['metrics']['description'] += 'Root Mean Squared Error of 2D automation vs 3D: %.2f\n'%rmse

            absolute_errors = np.abs(np.array(v2D_resampled) - np.array(v3D_DTW_resampled))
            standard_deviation = float(np.std(absolute_errors))
            cur_results_automation['metrics']['std_ae'] = standard_deviation
            cur_results_automation['metrics']['description'] += 'Standard Deviation of Absolute Errors: %.2f\n'%standard_deviation
            mean = float(np.mean(absolute_errors))
            assert mean == mae, 'Mean absolute error is not equal to mean of absolute errors'

            # Separate 2Dd and 3D automation means
            mean_2D = float(np.mean(v2D_resampled))
            mean_3D = float(np.mean(v3D_DTW_resampled))
            cur_results_automation['metrics']['mean_2D'] = mean_2D
            cur_results_automation['metrics']['mean_3D'] = mean_3D
            cur_results_automation['metrics']['description'] += 'Mean of 2D automation: %.2f\n'%mean_2D
            cur_results_automation['metrics']['description'] += 'Mean of 3D automation: %.2f\n'%mean_3D

            # Separate 2Dd and 3D automation std
            std_2D = float(np.std(v2D_resampled))
            std_3D = float(np.std(v3D_DTW_resampled))
            cur_results_automation['metrics']['std_2D'] = std_2D
            cur_results_automation['metrics']['std_3D'] = std_3D
            cur_results_automation['metrics']['description'] += 'Standard Deviation of 2D automation: %.2f\n'%std_2D
            cur_results_automation['metrics']['description'] += 'Standard Deviation of 3D automation: %.2f\n'%std_3D

            # Plot histogram of absolute errors
            # plt.figure(figsize=(18, 3))
            # plt.title('Histogram of Absolute Errors')
            # plt.hist(absolute_errors, bins=100)
            # plt.show()




            # print(cur_results_automation['metrics'])
            # pretty print dictionary of metrics
            # for k,v in cur_results_automation['metrics'].items():
            #     print('                 ',k,v)


        # plt.show()
        # exit()

        assert 'Recording Trajectory X / ControlGris' in cur_results_track.keys()
        assert 'Recording Trajectory Y / ControlGris' in cur_results_track.keys()
        assert 'Recording Trajectory Z / ControlGris' in cur_results_track.keys()

        
        ### Compute motion metrics

        #1. Displacement at each timestep
        
        for type in EXPERIMENT_TYPES:
            # print('\n###\n%s\n###'%type)
            x = np.array([float(e) for e in cur_results_track['Recording Trajectory X / ControlGris'][type]])
            y = np.array([float(e) for e in cur_results_track['Recording Trajectory Y / ControlGris'][type]])
            z = np.array([float(e) for e in cur_results_track['Recording Trajectory Z / ControlGris'][type]])
            
            # Number of time steps
            N = len(x)
            assert N == 1000
            assert N == len(y) == len(z), 'Different lengths for x, y, z'

            delta_t = ENDFILE_SECONDS / N

            # 1. Displacement at each time step ti
            displacements = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)


            # 2. Velocity at each time step ti
            velocities = displacements / delta_t

            # 3. Acceleration at each time step ti
            accelerations = (velocities[1:] - velocities[:-1]) / delta_t

            # 4. Total Quantity of Motion (QoM) over N time steps
            QoM = np.sum(displacements)

            # 5. Total Quantity of Velocity (QoV) over N time steps
            QoV = np.sum(velocities)

            # 6. Total Quantity of Acceleration (QoA) over N time steps
            QoA = np.sum(accelerations)

            # Output the calculated quantities
            # print(f"Total Quantity of Motion (QoM): {QoM}")
            # print(f"Total Quantity of Velocity (QoV): {QoV}")
            # print(f"Total Quantity of Acceleration (QoA): {QoA}")

            cur_results_track[type] = {}
            cur_results_track[type]['QoM'] = QoM
            cur_results_track[type]['QoV'] = QoV
            cur_results_track[type]['QoA'] = QoA

    for type in EXPERIMENT_TYPES:
        print('  Experiment type',type)
        
        typeQoMs = [cur_results[track][type]['QoM'] for track in all_tracks]
        typeQoVs = [cur_results[track][type]['QoV'] for track in all_tracks]
        typeQoAs = [cur_results[track][type]['QoA'] for track in all_tracks]

        # Averages
        avgQoM = np.mean(typeQoMs)
        avgQoV = np.mean(typeQoVs)
        avgQoA = np.mean(typeQoAs)

        # Standard deviations
        stdQoM = np.std(typeQoMs)
        stdQoV = np.std(typeQoVs)
        stdQoA = np.std(typeQoAs)

        cur_results['averages'] = {}
        cur_results['averages'][type] = {}
        cur_results['averages'][type]['QoM'] = avgQoM
        cur_results['averages'][type]['QoV'] = avgQoV
        cur_results['averages'][type]['QoA'] = avgQoA
        



for participant in PARTICIPANTS:
    print('Participant',participant)
    for track in results[participant]:
        print('  Track',track)
        for automation in results[participant][track]:
            if 'metrics' in results[participant][track][automation].keys():
                print('    Automation',automation)
                print('      Metrics:',results[participant][track][automation]['metrics']['description'].replace('\n','\n        '))

# save results as pickle
with open('results_and_resampled_data.pickle', 'wb') as f:
    pickle.dump(results, f)

            
if not DO_PLOT_DTW_PARTIAL:
    print('The maximum time shift applied by DTW was:',max_dtw_shift,'seconds')
    print('Info:',max_dtw_shift_info)