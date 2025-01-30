import argparse
import os
import os.path as op
import numpy as np
import pyxdf

import mne
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from mne_bids.stats import count_events

import json
import pandas as pd


def get_EyeTrackerChannels(cfg,info):
    channels = info["desc"][0]['channels'][0]["channel"]
    cfg['Columns'] = {}
    for c in channels:
        label = c['label'][0]
        cfg['Columns'][label] = {}
        cfg['Columns'][label]['LongName'] = "To determined" ## A changer
        cfg['Columns'][label]['Description'] = "To determined"  ## A changer
        cfg['Columns'][label]['units'] = c['unit']

    return cfg

def get_FlightSimuChannels(cfg,metadata):
    channels = metadata['Nom'].values
    cfg['Columns'] = {}
    for i in range(channels.shape[0]):
        label = channels[i]
        cfg['Columns'][label] = {}
        cfg['Columns'][label]['LongName'] = label ## A changer
        cfg['Columns'][label]['Description'] = metadata['Tables Description'].iloc[i]  ## A changer
        cfg['Columns'][label]['units'] = metadata['Unite'].iloc[i]

    return cfg

def get_FlightSimuMetricsChannels(cfg,info):
    channels = "to determined" #info["desc"][0]['channels'][0]["channel"]
    cfg['Columns'] = {}
    for c in channels:
        label = c['label'][0]
        cfg['Columns'][label] = {}
        cfg['Columns'][label]['LongName'] = "To determined" ## A changer
        cfg['Columns'][label]['Description'] = "To determined"  ## A changer
        cfg['Columns'][label]['units'] = c['unit']

    return cfg


def create_general_config(bids_root):
    dataset_description_cfg = {}
    dataset_description_cfg['Name'] = 'PROTEUS_BCI_Toulouse'
    dataset_description_cfg['BIDSVersion'] = "1.7.0"
    dataset_description_cfg['DatasetType'] = 'raw'
    dataset_description_cfg['License'] = 'PD'
    dataset_description_cfg['Authors'] = ['Cimarosto Pietro', 'Cabrera-Castillos Kalou', 'Velut Sebastien', 'Torre-Tresol Juan Jesus', 'Gomel Jules', 'Dehais Frederic']
    dataset_description_cfg['Funding'] = ['PROTEUS ANR']
    dataset_description_cfg['EthicsApprovals'] = ["Comite d'Ethique pour les Recherches de l'Universite de Toulouse"]
    dataset_description_cfg['DatasetDOI'] = "To Update"
    dataset_description_cfg['GeneratedBy'] = [{}]
    dataset_description_cfg['GeneratedBy'][0]['Name'] = "Velut Sebastien"
    dataset_description_cfg['GeneratedBy'][0]["Version"] = '1.0.0'
    dataset_description_cfg['GeneratedBy'][0]['Description'] = 'Manual and with BIDS MNE'

    # Convert and write JSON object to file
    if not os.path.exists(bids_root): 
        os.makedirs(bids_root) 
    with open(op.join(bids_root,"dataset_description.json"), "w") as outfile: 
        json.dump(dataset_description_cfg, outfile,indent=3)
    

def main(path,bids_root,participant,session,run,task,simu):
    simu = eval(simu)
    simuprefix = "simu" if simu else 'lab'
    #### GET DATA
    streams, header = pyxdf.load_xdf(op.join(path+participant+"/", '_'.join([participant,session, run,task+".xdf"])))
    keys = {''.join([j for j in streams[i]['info']['name'][0] if not j.isdigit() and j!="-"]):i for i in range(len(streams))}

    ############# EEG DATA #############

    ### Get EEG Data
    if task=='calib':
        # create channel of the flickers stim and trials
        flicker_stream = streams[keys["calibrationFlicker"]]
    elif task=='matb':
        flicker_stream = streams[keys['FoT']]
        print(len(flicker_stream["time_series"]))

    eeg_data = streams[keys['LiveAmpSN']]["time_series"].T
    sfreq = float(streams[keys['LiveAmpSN']]["info"]["nominal_srate"][0])
    channels_info = streams[keys['LiveAmpSN']]["info"]['desc'][0]["channels"][0]["channel"]
    channels = list(map(lambda x : x['label'][0], channels_info))
    info = mne.create_info(channels, sfreq, "eeg")
    raw = mne.io.RawArray(eeg_data*1e-6, info)
    raw.info["line_freq"] = 50  # specify power line frequency as required by BIDS

    ### Trasnform data for mne data
    if task=='calib':
        code_ind = np.array(flicker_stream['time_series'])[:,0] == 'TrialCode'
        bits_ind = np.array(flicker_stream['time_series'])[:,0] == 'Flicker'

        code = [d[1] for d in np.array(flicker_stream['time_series'])[code_ind]]
        bits = [d[1] for d in np.array(flicker_stream['time_series'])[bits_ind]]
        code_time = np.array(flicker_stream['time_stamps'])[code_ind]
        bits_time = np.array(flicker_stream['time_stamps'])[bits_ind]
    elif task=='matb':
        #Create the code XXX for each time stamp thank to the three by three timeseries
        nb_point = np.array(flicker_stream['time_series']).shape[0]
        bits = [flicker_stream['time_series'][i][2] + flicker_stream['time_series'][i+1][2] + flicker_stream['time_series'][i+2][2] for i in range(0,nb_point,3)]
        bits_time = [flicker_stream['time_stamps'][i] for i in range(0,nb_point,3)]

    eeg_start = streams[keys['LiveAmpSN']]['time_stamps'][0]
    stim_chan = np.zeros((1, len(raw)))
    stim_onset = np.zeros(len(bits))
    if task=='calib':
        trial_onset = np.zeros(len(code))
        trial_chan = np.zeros((1, len(raw)))
    stim_label = []
    trial_label = []
    i=0
    for onset, label in zip(bits_time, bits):
        onset_frame = int((onset-eeg_start)*sfreq)
        stim_chan[0, onset_frame] = label
        stim_onset[i] = (onset-eeg_start)
        stim_label.append(label)
        i+=1

    if task=='calib':
        i=0
        for onset, label in zip(code_time, code):
            onset_frame = int((onset-eeg_start)*sfreq)
            trial_chan[0, onset_frame] = label
            trial_onset[i] = (onset-eeg_start)
            trial_label.append(label)
            i+=1

    anno = mne.Annotations(stim_onset,1/60,np.array(stim_label))
    if task=='calib':
        anno.append(trial_onset,10,np.array(trial_label))
    raw_data = raw.set_annotations(anno)

    ### Convert to BIDS
    bids_path = BIDSPath(subject=participant, task=task+simuprefix, session=str(session), run=str(run), root=bids_root)
    write_raw_bids(raw_data, bids_path, overwrite=True, allow_preload=True, format="EEGLAB")

    ### Change in config
    eeg_cfg = {}
    eeg_cfg['TaskName'] = task+simuprefix
    eeg_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
    eeg_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
    eeg_cfg['Manufacturer'] = "LiveAmp" ## A changer
    eeg_cfg['ManufacturerModelName'] = "LiveAMP" ## A changer
    eeg_cfg['DeviceSerialNumber'] = "To determined" ## A changer
    eeg_cfg['SoftwareVersions'] = "Enobio" ## A changer
    eeg_cfg['InstitutionName'] = 'ISAE-SUPAERO'
    eeg_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pelegrin, BP 54032, 31055 Toulouse Cedex 4 France"
    eeg_cfg['InstitutionalDepartmentName'] = "CNE"

    eeg_cfg['RecordingDuration'] = bits_time[-1]-bits_time[0]
    eeg_cfg['RecordingType'] = 'continuous'
    eeg_cfg['EEGReference'] = 'Cz'
    eeg_cfg['SamplingFrequency'] = sfreq
    eeg_cfg['PowerLineFrequency'] = 50.0
    eeg_cfg['SoftwareFilters'] = 'n/a'
    eeg_cfg['CapManufacturer'] = 'Enobio'
    eeg_cfg['CapManufacturersModelName'] = 'n/a'
    eeg_cfg['EEGPlacementScheme'] = "based on the extended 10/20 system"
    eeg_cfg['EEGChannelCount'] = 32
    eeg_cfg['ECGChannelCount'] = 0
    eeg_cfg['EOGChannelCount'] = 0
    eeg_cfg['EMGChannelCount'] = 0
    eeg_cfg['MiscChannelCount'] = 0
    eeg_cfg['TriggerChannelCount'] = 0


    eeg_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"eeg")
    eeg_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_eeg.json"
    if not os.path.exists(eeg_path): 
        os.makedirs(eeg_path) 
    with open(op.join(eeg_path,eeg_cfg_name), "w") as outfile: 
        json.dump(eeg_cfg, outfile,indent=3)

    ############# EyeTracker DATA #############
    ### Get Data
    et_stream = streams[keys['GazepointEyeTracker']]
    et_info = et_stream['info']
    et_data = et_stream['time_series']
    et_time = et_stream['time_stamps']

    ### Create JSON
    et_cfg = {}
    et_cfg["SamplingFrequency"] = et_info['effective_srate']
    et_cfg["NominalSamplingFrequency"] = et_info['nominal_srate']
    et_cfg['StartTime'] = et_time[0]
    et_cfg["BIDSVersion"] = 'n/a'
    et_cfg['TaskName'] = task+simuprefix
    et_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
    et_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
    et_cfg['Manufacturer'] = "To determined" ## A changer
    et_cfg['ManufacturerModelName'] = "To determined" ## A changer
    et_cfg['DeviceSerialNumber'] = "To determined" ## A changer
    et_cfg['SoftwareVersions'] = "To determined" ## A changer
    et_cfg['InstitutionName'] = 'ISAE-SUPAERO'
    et_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
    et_cfg['InstitutionalDepartmentName'] = "CNE"

    ### Channel in the generic eye tracker json
    et_cfg = get_EyeTrackerChannels(et_cfg,et_info)

    ### Create data tsv
    n_channels = et_data.shape[1]
    et_header = [et_info["desc"][0]['channels'][0]["channel"][i]['label'][0] for i in range(n_channels)]
    et_pd = pd.DataFrame(et_data,columns=et_header)

    ### save file
    et_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"beh")
    et_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_eyetracker.tsv"
    et_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_eyetracker.json"

    if not os.path.exists(et_path): 
        os.makedirs(et_path) 
    et_pd.to_csv(op.join(et_path,et_data_name),sep='\t')
    # Convert and write JSON object to file
    with open(op.join(et_path,et_cfg_name), "w") as outfile: 
        json.dump(et_cfg, outfile,indent=3)

    ############# Flight Simulator Data #############
    ### Get Data
    if simu and task=='matb':
        fs_stream = streams[keys['PEGASE_Data']]
        fs_info = fs_stream['info']
        fs_data = fs_stream['time_series']
        fs_time = fs_stream['time_stamps']

        fsm_stream = streams[keys['Flying_metrics']]
        fsm_info = fsm_stream['info']
        fsm_data = fsm_stream['time_series']
        fsm_time = fsm_stream['time_stamps']

        ### create metadata
        metadata_pegase = pd.read_csv(path+"pegase_metadata.csv",sep=';')

        ### Channel in the flight pegase data json
        fs_cfg = {}
        fs_cfg["SamplingFrequency"] = fs_info['effective_srate']
        fs_cfg["NominalSamplingFrequency"] = fs_info['nominal_srate']
        fs_cfg['StartTime'] = fs_time[0]
        fs_cfg["BIDSVersion"] = 'n/a'
        fs_cfg['TaskName'] = task
        fs_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
        fs_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
        fs_cfg['Manufacturer'] = "ISAE-SUPAERO" ## A changer
        fs_cfg['ManufacturerModelName'] = "PEGASE" ## A changer
        fs_cfg['DeviceSerialNumber'] = "None" ## A changer
        fs_cfg['SoftwareVersions'] = "PEGASE" ## A changer
        fs_cfg['InstitutionName'] = 'ISAE-SUPAERO'
        fs_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
        fs_cfg['InstitutionalDepartmentName'] = "CNE"
        fs_cfg = get_FlightSimuChannels(fs_cfg,metadata_pegase)

        ### Channel in the flight metrics json
        fsm_cfg = {}
        fsm_cfg["SamplingFrequency"] = fsm_info['effective_srate']
        fsm_cfg["NominalSamplingFrequency"] = fsm_info['nominal_srate']
        fsm_cfg['StartTime'] = fsm_time[0]
        fsm_cfg["BIDSVersion"] = 'n/a'
        fsm_cfg['TaskName'] = task
        fsm_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
        fsm_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
        fsm_cfg['Manufacturer'] = "ISAE-SUPAERO" ## A changer
        fsm_cfg['ManufacturerModelName'] = "PEGASE" ## A changer
        fsm_cfg['DeviceSerialNumber'] = "None" ## A changer
        fsm_cfg['SoftwareVersions'] = "PEGASE" ## A changer
        fsm_cfg['InstitutionName'] = 'ISAE-SUPAERO'
        fsm_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
        fsm_cfg['InstitutionalDepartmentName'] = "CNE"
        ### Channel in the generic eye tracker json
        fsm_cfg['Columns'] = {}
        fsm_cfg['Columns']['iWaypoint'] = {}
        fsm_cfg['Columns']['iWaypoint']['LongName'] = "Waypoint Index"
        fsm_cfg['Columns']['iWaypoint']['Description'] = "Index of the waypoint sent to the simulator"
        fsm_cfg['Columns']['Dist'] = {}
        fsm_cfg['Columns']['Dist']['LongName'] = "Distance"
        fsm_cfg['Columns']['Dist']['Description'] = "Distance to the next waypoint" 
        fsm_cfg['Columns']['Dist']['units'] = "meters"
        fsm_cfg['Columns']['Heading'] = {}
        fsm_cfg['Columns']['Heading']['LongName'] = "Heading"
        fsm_cfg['Columns']['Heading']['Description'] = "Angle between the plane and the net waypoint" 
        fsm_cfg['Columns']['Heading']['units'] = "degree between -180 and 180" 
        fsm_cfg['Columns']['Perf'] = {}
        fsm_cfg['Columns']['Perf']['LongName'] = "Performance"
        fsm_cfg['Columns']['Perf']['Description'] = "Performance of the user to reach the wanted waypoint" 
        fsm_cfg['Columns']['Altitude_visee'] = {}
        fsm_cfg['Columns']['Altitude_visee']['LongName'] = "Altitude_visee"
        fsm_cfg['Columns']['Altitude_visee']['Description'] = "Altitude of the next waypoint"
        fsm_cfg['Columns']['Vitesse'] = {}
        fsm_cfg['Columns']['Vitesse']['LongName'] = "Plane_Speed"
        fsm_cfg['Columns']['Vitesse']['Description'] = "Speed of the plane" 

        ### Create tsv data
        fs_header = metadata_pegase['Nom'].values
        fs_pd = pd.DataFrame(fs_data,columns=fs_header)

        fsm_header = ["iWaypoint", "Dist", "Heading", "Perf", "Altitude_visee", "Vitesse"]
        fsm_pd = pd.DataFrame(fsm_data,columns=fsm_header)

        ### save files
        fs_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"fsimu")
        fs_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_flightsimulatordata.tsv"
        fs_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_flightsimulatordata.json"
        if not os.path.exists(fs_path): 
            os.makedirs(fs_path) 
        fs_pd.to_csv(op.join(fs_path,fs_data_name),sep='\t')
        # Convert and write JSON object to file
        with open(op.join(fs_path,fs_cfg_name), "w") as outfile: 
            json.dump(fs_cfg, outfile,indent=3)

        ### path to save Flying Metrics
        fsm_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"fsimu")
        fsm_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_flyingmetrics.tsv"
        fsm_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_flyingmetrics.json"
        if not os.path.exists(fsm_path): 
            os.makedirs(fsm_path) 
        fsm_pd.to_csv(op.join(fsm_path,fsm_data_name),sep='\t')
        # Convert and write JSON object to file
        with open(op.join(fsm_path,fsm_cfg_name), "w") as outfile: 
            json.dump(fsm_cfg, outfile,indent=3)

    ############# Extra Data #############
    ### Get data
    if task=='matb':
        foc_stream = streams[keys['Focus']]
        foc_info = foc_stream['info']
        foc_data = foc_stream['time_series']
        foc_time = foc_stream['time_stamps']

        matb_stream = streams[keys['MATB']]
        matb_info = matb_stream['info']
        matb_data = matb_stream['time_series']
        matb_time = matb_stream['time_stamps']

        pred_stream = streams[keys['Prediction']]
        pred_info = pred_stream['info']
        pred_data = pred_stream['time_series']
        pred_time = pred_stream['time_stamps']

        ### Create config
        pred_cfg = {}
        pred_cfg["SamplingFrequency"] = pred_info['effective_srate']
        pred_cfg["NominalSamplingFrequency"] = pred_info['nominal_srate']
        pred_cfg['StartTime'] = pred_time[0]
        pred_cfg["BIDSVersion"] = 'n/a'
        pred_cfg['TaskName'] = task
        pred_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
        pred_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
        pred_cfg['Manufacturer'] = "ISAE-SUPAERO" ## A changer
        pred_cfg['ManufacturerModelName'] = "NONE" ## A changer
        pred_cfg['DeviceSerialNumber'] = "None" ## A changer
        pred_cfg['SoftwareVersions'] = "NONE" ## A changer
        pred_cfg['InstitutionName'] = 'ISAE-SUPAERO'
        pred_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
        pred_cfg['InstitutionalDepartmentName'] = "CNE"
        pred_cfg['DataDescription'] = "Prediction of the online classifier to better understand what the participant saw for feedback"
        ## Put a value for describing what is the data recorded there

        pred_cfg['Columns'] = {}
        pred_cfg['Columns']['rate'] = {}
        pred_cfg['Columns']['rate']['LongName'] = "Sampling rate"
        pred_cfg['Columns']['rate']['Description'] = "Sampling rate of the data trained on" 
        pred_cfg['Columns']['rate']['units'] = 'Hz'
        pred_cfg['Columns']['onset'] = {}
        pred_cfg['Columns']['onset']['LongName'] = "Epoch Onset"
        pred_cfg['Columns']['onset']['Description'] = "Onset of the epoch predicted" 
        pred_cfg['Columns']['onset']['units'] = "Datetime AAAA-MM-DD HH:MM:SS"
        pred_cfg['Columns']['cIndex'] = {}
        pred_cfg['Columns']['cIndex']['LongName'] = "Context Frame index"
        pred_cfg['Columns']['cIndex']['Description'] = "Index of the frame of the prediction since the beginning" 
        pred_cfg['Columns']['cBits'] = {}
        pred_cfg['Columns']['cBits']['LongName'] = "Context Bits"
        pred_cfg['Columns']['cBits']['Description'] = "state of the different flicker on the frame" 
        pred_cfg['Columns']['results'] = {}
        pred_cfg['Columns']['results']['LongName'] = "Results"
        pred_cfg['Columns']['results']['Description'] = "Results of the online classifier" 

        matb_cfg = {}
        matb_cfg["SamplingFrequency"] = matb_info['effective_srate']
        matb_cfg["NominalSamplingFrequency"] = matb_info['nominal_srate']
        matb_cfg['StartTime'] = matb_time[0]
        matb_cfg["BIDSVersion"] = 'n/a'
        matb_cfg['TaskName'] = task
        matb_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
        matb_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
        matb_cfg['Manufacturer'] = "ISAE-SUPAERO" ## A changer
        matb_cfg['ManufacturerModelName'] = "NONE" ## A changer
        matb_cfg['DeviceSerialNumber'] = "None" ## A changer
        matb_cfg['SoftwareVersions'] = "NONE" ## A changer
        matb_cfg['InstitutionName'] = 'ISAE-SUPAERO'
        matb_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
        matb_cfg['InstitutionalDepartmentName'] = "CNE"
        matb_cfg['DataDescription'] = 'Events and information from the MATB task performed by the participant'

        matb_cfg['Columns'] = {}
        matb_cfg['Columns']['time'] = {}
        matb_cfg['Columns']['time']['LongName'] = "Time"
        matb_cfg['Columns']['time']['Description'] = "Time of the event since the start of the application" 
        matb_cfg['Columns']['time']['units'] = 'seconds'
        matb_cfg['Columns']['label'] = {}
        matb_cfg['Columns']['label']['LongName'] = "Event label"
        matb_cfg['Columns']['label']['Description'] = "label of the event sent by the application" 
        matb_cfg['Columns']['value'] = {}
        matb_cfg['Columns']['value']['LongName'] = "Event value"
        matb_cfg['Columns']['value']['Description'] = "value of the associated event sent by the application" 

        foc_cfg = {}
        foc_cfg["SamplingFrequency"] = foc_info['effective_srate']
        foc_cfg["NominalSamplingFrequency"] = foc_info['nominal_srate']
        foc_cfg['StartTime'] = foc_time[0]
        foc_cfg["BIDSVersion"] = 'n/a'
        foc_cfg['TaskName'] = task
        foc_cfg['TaskDescription'] = "Calibration before main experiment" if task=="calib" else "Main experiment by performing MATB task"
        foc_cfg['Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task=="calib" else "Perform the 3 task of the MATB"
        foc_cfg['Manufacturer'] = "ISAE-SUPAERO" ## A changer
        foc_cfg['ManufacturerModelName'] = "NONE" ## A changer
        foc_cfg['DeviceSerialNumber'] = "None" ## A changer
        foc_cfg['SoftwareVersions'] = "NONE" ## A changer
        foc_cfg['InstitutionName'] = 'ISAE-SUPAERO'
        foc_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
        foc_cfg['InstitutionalDepartmentName'] = "CNE"
        foc_cfg['DataDescription'] = 'Prediction of the focus of the participant after accumulation'

        ### create tsv 
        pred_header = ['rate','onset','cIndex','cBits','results']
        n_values = len(pred_data)
        dict_list = [eval(pred_data[i][0].replace('Timestamp','')) for i in range(n_values)]
        pred_data = [[dict_list[i]['rate'], dict_list[i]['epoch']['onset'],dict_list[i]['epoch']['context']['index'],dict_list[i]['epoch']['context']['bits'],eval(pred_data[i][1])['result']] for i in range(n_values)]
        pred_pd = pd.DataFrame(pred_data,columns=pred_header)

        matb_header = ['time','label','value']
        matb_pd = pd.DataFrame(matb_data,columns=matb_header)

        foc_pd = pd.DataFrame(foc_data,columns=["focus"])

        ### save file
        pred_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"extradata")
        pred_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_predictiondata.tsv"
        pred_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_predictiondata.json"
        if not os.path.exists(pred_path): 
            os.makedirs(pred_path) 
        pred_pd.to_csv(op.join(pred_path,pred_data_name),sep='\t')
        # Convert and write JSON object to file
        with open(op.join(pred_path,pred_cfg_name), "w") as outfile: 
            json.dump(pred_cfg, outfile,indent=3)


        ### path to save MATB data
        matb_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"extradata")
        matb_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_matbdata.tsv"
        matb_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_matbdata.json"
        if not os.path.exists(matb_path): 
            os.makedirs(matb_path) 
        matb_pd.to_csv(op.join(matb_path,matb_data_name),sep='\t')
        # Convert and write JSON object to file
        with open(op.join(matb_path,matb_cfg_name), "w") as outfile: 
            json.dump(matb_cfg, outfile,indent=3)

        ### path to save Focus data
        foc_path = op.join(bids_root,"sub-"+participant,"ses-"+session,"extradata")
        foc_data_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_focusdata.tsv"
        foc_cfg_name = "sub-"+participant+"_ses-"+session+"_task-"+task+simuprefix+"_run-"+run+"_focusdata.json"
        if not os.path.exists(foc_path): 
            os.makedirs(foc_path) 
        foc_pd.to_csv(op.join(foc_path,foc_data_name),sep='\t')
        # Convert and write JSON object to file
        with open(op.join(foc_path,foc_cfg_name), "w") as outfile: 
            json.dump(foc_cfg, outfile,indent=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the data to tranform")
    parser.add_argument("--bids_root", help="Patht to folder being in BIDS format")
    parser.add_argument("--participant",help="Participant id (ex: s12) which data we want to transform in BIDS")
    parser.add_argument("--session",default="1",help="session of the data to transform in BIDS")
    parser.add_argument("--run",default="1",help="run of the data to transform in BIDS")
    parser.add_argument("--task",help="task of the data to transform in BIDS (only 'calib' or 'matb')")
    parser.add_argument("--simu",help="Boolean to know if it was in the simulator or not")
    parser.add_argument("--general_config",default='False',help="Boolean to know if you want to create the general config or not")

    args = parser.parse_args()
    general_config = eval(args.general_config)
    print(general_config)
    if general_config:
        create_general_config(args.bids_root)
    else:
        main(args.path,args.bids_root,args.participant,args.session,args.run,args.task,args.simu)
