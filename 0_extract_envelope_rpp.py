import os, numpy as np, re
from glob import glob



automation_dict = {}

VERBOSE = False
printVerbose = print if VERBOSE else lambda *a, **k: None



def extract_from_rpp_project(rpp_filename):

    with open(rpp_filename, 'r') as f:
        lines = f.readlines()
        
        readname = False
        readAutomation = None
        trackname = None
        for line in lines:
            line = line.rstrip('\n')
            if '<TRACK' in line:
                readname = True


                # if trackname is not None:
                #     curauto = automation_dict[trackname]
                #     for i in range(len(curauto)):
                #         # print('keys:',list(curauto[i].keys()))
                #         assert curauto[i]['index'] == i, 'Index error ({} != {})'.format(curauto[i]['index'], i)
                #         # print('\t\t>>> automation name',curauto[i]['name'])
                #         # print('\t\t>>> track name',curauto[i]['track'])
                #         # print('\t\t>>> len(automation)',len(curauto[i]['automation']))
                #         # print('\t\t\tlen(curauto[i][\'automation\'])',len(curauto[i]['automation']))
                #         assert 'automation' in curauto[i], 'No automation in track {}'.format(trackname)

                #         print('\tAutomation #%d \"%s\" has %d points'%(i,curauto[i]['name'],len(curauto[i]['automation'])))


            if readname and 'NAME' in line:
                trackname = line.lstrip('    NAME "').rstrip('"\n')
                printVerbose('found Track',trackname)
                automation_dict[trackname] = list()
                readname = False
                readAutomation = -1

            if readAutomation is not None and '<PARMENV' in line:
                index = re.search(r'<PARMENV (\d+)', line).group(1)
                name = re.search(r'"([^"]+)"', line).group(1)
                printVerbose('Index: "'+index+'"')
                printVerbose('Name: "'+name+'"')

                automation_dict[trackname].append({'index': int(index),'name': name, 'track':trackname})
                printVerbose('Appending automation',name, 'to track',trackname)
                for i in range(len(automation_dict[trackname])):
                    assert automation_dict[trackname][i]['index'] == i, 'Index error ({} != {})'.format(automation_dict[trackname][i]['index'], i)

                readAutomation += 1
                if readAutomation == 4:
                    readAutomation = None
            elif readAutomation is not None and readAutomation>-1 and ' PT ' in line:
                printVerbose(line)
                # Get PT ([0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*)
                vals = re.search(r'PT ([0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*) ([0-9]+\.?[0-9]*)', line)
                time = float(vals.group(1))
                value = float(vals.group(2))
                shape = float(vals.group(3))
                printVerbose ('Time: {}, Value: {}, Shape: {}'.format(time, value, shape))
                assert shape == 0 or shape==1, 'Shape at time {} of automation-track {}-{} not Linear (0) but {}'.format(time,
                                                                                                            automation_dict[trackname][readAutomation]['name'],
                                                                                                            trackname,
                                                                                                            shape)
                if 'automation' not in automation_dict[trackname][readAutomation]:
                    automation_dict[trackname][readAutomation]['automation'] = []
                automation_dict[trackname][readAutomation]['automation'].append((time,value,shape))


    for trackname in automation_dict:
        curauto = automation_dict[trackname]    
        print('Track',trackname)
        assert len(curauto) <= 3, 'Track {} of file {} has {} automations, not <3'.format(trackname, os.path.basename(rpp_filename), len(curauto))
        for i in range(len(curauto)):
            assert curauto[i]['index'] == i, 'Index error ({} != {})'.format(curauto[i]['index'], i)
            assert 'automation' in curauto[i], 'No automation in track {}'.format(trackname)
            print('\tAutomation #%d \"%s\" has %d points'%(i,curauto[i]['name'],len(curauto[i]['automation'])))


    return automation_dict

# # import json
# import pickle
# with open('automation%s.pickle'%os.path.splitext(os.path.basename(rpp_filename))[0], 'wb') as f:
#     pickle.dump(automation_dict, f)




rpp_files = sorted(glob('./Reaper projects/*/*.rpp'))
assert len(rpp_files) == 24, 'Expected 24 rpp files, found {}'.format(len(rpp_files))


per_participant_automation_dict = {}
for rpp_filename in rpp_files:
    print('Processing',rpp_filename)
    automation_dict = extract_from_rpp_project(rpp_filename)
    

    # Extract IDx from filename
    participant_id = os.path.basename(rpp_filename).split('_')[0]
    print('Participant ID:',participant_id)

    if participant_id not in per_participant_automation_dict:
        per_participant_automation_dict[participant_id] = {}

    if '_2DReaper' in rpp_filename or '_Reaper2D' in rpp_filename:
        per_participant_automation_dict[participant_id]['2D'] = automation_dict.copy()
    elif '_3DUnity' in rpp_filename:
        per_participant_automation_dict[participant_id]['3D'] = automation_dict.copy()
    else:
        raise ValueError('Unknown file type:',rpp_filename)

import pickle
with open('extracted_envelopes.pickle', 'wb') as f:
    pickle.dump(per_participant_automation_dict, f)
    