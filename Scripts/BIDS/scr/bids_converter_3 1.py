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

# def save_stream(streams, key, header, path, savename, cfg):
#     if key in keys:
#         stream = streams[keys[key]]
#         info = stream['info']
#         data = stream['time_series']
#         time = stream['time_stamps']
#
#         df = pd.DataFrame(data, columns=header)
#
#         ### path to save MATB data
#         matb_path = op.join(bids_root, "sub-" + participant, "ses-" + session, "extradata")
#         data_name = savename + "_matbdata.tsv"
#         cfg_name = savename + "_matbdata.json"
#         if not os.path.exists(path):
#             os.makedirs(path)
#         df.to_csv(op.join(path, data_name), sep='\t')
#         # Convert and write JSON object to file
#         with open(op.join(path, cfg_name), "w") as outfile:
#             json.dump(cfg, outfile, indent=3)
#
#     return info

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


def create_subj_config(bids_root):
    subj_cfg = {}

    subj_cfg['Participant'] = {}
    subj_cfg['Participant']['Description'] = "Unique participant identifier"
    subj_cfg['Age'] = {}
    subj_cfg['Age']['Description'] = "Unique participant identifier"
    subj_cfg['Age']['Units'] = "Years"
    subj_cfg['Sex'] = {}
    subj_cfg['Sex']['Description'] = "Biological sex of the participant"
    subj_cfg['Sex']['Answer'] = {}
    subj_cfg['Sex']['Answer']['M'] = "Male"
    subj_cfg['Sex']['Answer']['F'] = "Female"
    subj_cfg['Hand'] = {}
    subj_cfg['Hand']['Description'] = "Handedness of the participant"
    subj_cfg['Hand']['Answer'] = {}
    subj_cfg['Hand']['Answer']['R'] = "Right"
    subj_cfg['Hand']['Answer']['L'] = "Left"
    subj_cfg['Education'] = {}
    subj_cfg['Education']['Description'] = "Level of scholastic education of the participant"

    subj_cfg['VideogameQ1'] = {}
    subj_cfg['VideogameQ1'][
        'Description'] = "Number of times the participant launched a videogames in the last 12 months"
    subj_cfg['VideogameQ1']['Answer'] = {}
    subj_cfg['VideogameQ1']['Answer']['0'] = "0 times"
    subj_cfg['VideogameQ1']['Answer']['1-10'] = "1 to 10 times (~1/month)"
    subj_cfg['VideogameQ1']['Answer']['11-30'] = "11 to 30 times (~2/month)"
    subj_cfg['VideogameQ1']['Answer']['31-60'] = "31 to 60 times (3 to 5/month)"
    subj_cfg['VideogameQ1']['Answer']['60+'] = "more than 60 times"
    subj_cfg['VideogameQ2'] = {}
    subj_cfg['VideogameQ2']['Description'] = "Average hours per day the participant spends playing videogames"
    subj_cfg['VideogameQ2']['Answer'] = {}
    subj_cfg['VideogameQ2']['Answer']['0'] = "0 minutes"
    subj_cfg['VideogameQ2']['Answer']['0-30'] = "0 to 30 minutes"
    subj_cfg['VideogameQ2']['Answer']['30-60'] = "30 to 60 minutes"
    subj_cfg['VideogameQ2']['Answer']['1-2'] = "1 to 2 hours"
    subj_cfg['VideogameQ2']['Answer']['2-3'] = "2 to 3 hours"
    subj_cfg['VideogameQ2']['Answer']['3-4'] = "3 to 4 hours"
    subj_cfg['VideogameQ2']['Answer']['4+'] = "More than 4 hours"
    subj_cfg['VideogameQ3'] = {}
    subj_cfg['VideogameQ3']['Description'] = "The demographics.considers him/herself an active videogames user"
    subj_cfg['VideogameQ3']['Answer'] = {}
    subj_cfg['VideogameQ3']['Answer']['Y'] = "Yes"
    subj_cfg['VideogameQ3']['Answer']['N'] = "No"
    subj_cfg['VideogameQ4'] = {}
    subj_cfg['VideogameQ4'][
        'Description'] = "The demographics.has had periods of life where he/her played videogames, on average, more than 2 hours per day"
    subj_cfg['VideogameQ4']['Answer'] = {}
    subj_cfg['VideogameQ4']['Answer']['Y'] = "Yes"
    subj_cfg['VideogameQ4']['Answer']['N'] = "No"

    subj_cfg['Flight_Exp'] = {}
    subj_cfg['Flight_Exp']['Description'] = "Experience with flight-based tasks and piloting"
    subj_cfg['Flight_Exp']['Answer'] = {}
    subj_cfg['Flight_Exp']['Answer'][
        '0'] = "The participant had zero experience with flight-based tasks and had little or no substantial knowledge about flight dynamics."
    subj_cfg['Flight_Exp']['Answer'][
        '1'] = 'The participant was studying / studied flight dynamics but had little or no experience with flight simulation software or engaged in a realistic aircraft piloting task.'
    subj_cfg['Flight_Exp']['Answer'][
        '2'] = "The participant was training or was trained to fly and regularly used realistic flight simulator program and/or piloted real aircrafts."

    subj_cfg['DES'] = {}
    subj_cfg['DES'][
        'Description'] = "The participant rates how often you experience each ite as a score from 1 (never) to 5 (always)"
    subj_cfg['DES']['Question'] = {}
    subj_cfg['DES']['Question']['1'] = "Dry eyes (frequently feeling dryness in the eyes)"
    subj_cfg['DES']['Question']['2'] = "Eye strain (frequently feeling of pain or strain in the eyes)"
    subj_cfg['DES']['Question']['3'] = "Irritation or burning"
    subj_cfg['DES']['Question']['4'] = "Red eyes (without an apparent reason)"
    subj_cfg['DES']['Question']['5'] = "Photophobia (sensitivity to bright lights)"
    subj_cfg['DES']['Question']['6'] = "Halo (notice a ring of light surrounding objects)"
    subj_cfg['DES']['Question']['7'] = "Blurred vision"
    subj_cfg['DES']['Question']['8'] = "Feeling of foreign body in the eye (without an apparent reason)"
    subj_cfg['DES']['Question']['9'] = "Irritation / prickling sensation"
    subj_cfg['DES']['Question']['10'] = "Watery eyes (tears without an apparent reason"
    subj_cfg['DES']['Question']['11'] = "Diplopia (seeing double)"
    subj_cfg['DES']['Question']['12'] = "Headache (not related to injury ir migraine)"
    subj_cfg['DES']['Question']['13'] = "Shoulder and/or neck pain (without an apparent reason)"
    subj_cfg['DES']['Range'] = "0 (Never) to 5 (Always)"

    # Convert and write JSON object to file
    if not os.path.exists(bids_root):
        os.makedirs(bids_root)
    with open(op.join(bids_root, "demographics.json"), "w") as outfile:
        json.dump(subj_cfg, outfile, indent=3)


def create_session_config(bids_sessionpath):
    ses_cfg = {}

    ses_cfg['Session'] = {}
    ses_cfg['Session']['Description'] = "Session number"
    ses_cfg['Date'] = {}
    ses_cfg['Date']['Description'] = "Date of the session"
    ses_cfg['Hour'] = {}
    ses_cfg['Hour']['Description'] = "Time interval in which the session took place"
    ses_cfg['Hour']['Answer'] = {}
    ses_cfg['Hour']['Answer']['9-11'] = "9am to 11am"
    ses_cfg['Hour']['Answer']['13-15'] = "1pm to 3pm"
    ses_cfg['Hour']['Answer']['16-18'] = "4pm to 6pm"
    ses_cfg['VAS-F'] = {}
    ses_cfg['VAS-F'][
        'Description'] = "Visual Analog Scale to evaluate Fatigue Severity (VAS-F) - The participant has to score each item regarding his/her level of energy at the time of the experiment"
    ses_cfg['VAS-F']['Question'] = {}
    ses_cfg['VAS-F']['Question']['1'] = "Tired"
    ses_cfg['VAS-F']['Question']['2'] = "Sleepy"
    ses_cfg['VAS-F']['Question']['3'] = "Drowsy"
    ses_cfg['VAS-F']['Question']['4'] = "Fatigued"
    ses_cfg['VAS-F']['Question']['5'] = "Worn out"
    ses_cfg['VAS-F']['Question']['6'] = "Energetic"
    ses_cfg['VAS-F']['Question']['7'] = "Active"
    ses_cfg['VAS-F']['Question']['8'] = "Vigorous"
    ses_cfg['VAS-F']['Question']['9'] = "Efficient"
    ses_cfg['VAS-F']['Question']['10'] = "Lively"
    ses_cfg['VAS-F']['Question']['11'] = "Bushed"
    ses_cfg['VAS-F']['Question']['12'] = "Exhausted"
    ses_cfg['VAS-F']['Question']['13'] = "Keeping my eyes open"
    ses_cfg['VAS-F']['Question']['14'] = "Moving my body"
    ses_cfg['VAS-F']['Question']['15'] = "Concentrating"
    ses_cfg['VAS-F']['Question']['16'] = "Carrying on a conversation"
    ses_cfg['VAS-F']['Question']['17'] = "I have desire to close my eyes"
    ses_cfg['VAS-F']['Question']['18'] = "I have desire to lie down"
    ses_cfg['VAS-F']['Answer'] = {}
    ses_cfg['VAS-F']['Answer']['1_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['2_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['3_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['4_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['5_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['6_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['7_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['8_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['9_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['10_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['11_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['12_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['13_range3'] = "0 (no effort at all) to 10 (is a tremendous chore)"
    ses_cfg['VAS-F']['Answer']['14_range'] = "0 (no effort at all) to 10 (is a tremendous chore)"
    ses_cfg['VAS-F']['Answer']['15_range'] = "0 (no effort at all) to 10 (is a tremendous chore)"
    ses_cfg['VAS-F']['Answer']['16_range'] = "0 (no effort at all) to 10 (is a tremendous chore)"
    ses_cfg['VAS-F']['Answer']['17_range'] = "0 (not at all) to 10 (extremely)"
    ses_cfg['VAS-F']['Answer']['18_range'] = "0 (not at all) to 10 (extremely)"

    ses_cfg['Chalder_FS'] = {}
    ses_cfg['Chalder_FS'][
        'Description'] = "Chalder fatigue scale - The participant has to answer each question considering the last month as time interval of reference"
    ses_cfg['Chalder_FS']['Question'] = {}
    ses_cfg['Chalder_FS']['Question']['1'] = "Do you have problems with tiredness?"
    ses_cfg['Chalder_FS']['Question']['2'] = "Do you need to rest more?"
    ses_cfg['Chalder_FS']['Question']['3'] = "Do you feel sleepy or drowsy?"
    ses_cfg['Chalder_FS']['Question']['4'] = "Do you have problems starting things?"
    ses_cfg['Chalder_FS']['Question']['5'] = "Do you lack energy?"
    ses_cfg['Chalder_FS']['Question']['6'] = "Do you have less strength in your muscles?"
    ses_cfg['Chalder_FS']['Question']['7'] = "Do you feel weak?"
    ses_cfg['Chalder_FS']['Question']['8'] = "Do you have difficulties concentrating?"
    ses_cfg['Chalder_FS']['Question']['9'] = "Do you make slips of the tongue when speaking?"
    ses_cfg['Chalder_FS']['Question']['10'] = "Do you find it more difficult to find the right word?"
    ses_cfg['Chalder_FS']['Question']['11'] = "Do you have difficulties concentrating?"
    ses_cfg['Chalder_FS']['Answer'] = {}
    ses_cfg['Chalder_FS']['Answer']['1'] = {}
    ses_cfg['Chalder_FS']['Answer']['1']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['1']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['1']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['1']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['2'] = {}
    ses_cfg['Chalder_FS']['Answer']['2']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['2']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['2']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['2']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['3'] = {}
    ses_cfg['Chalder_FS']['Answer']['3']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['3']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['3']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['3']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['4'] = {}
    ses_cfg['Chalder_FS']['Answer']['4']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['4']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['4']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['4']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['5'] = {}
    ses_cfg['Chalder_FS']['Answer']['5']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['5']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['5']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['5']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['6'] = {}
    ses_cfg['Chalder_FS']['Answer']['6']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['6']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['6']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['6']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['7'] = {}
    ses_cfg['Chalder_FS']['Answer']['7']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['7']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['7']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['7']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['8'] = {}
    ses_cfg['Chalder_FS']['Answer']['8']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['8']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['8']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['8']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['9'] = {}
    ses_cfg['Chalder_FS']['Answer']['9']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['9']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['9']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['10'] = {}
    ses_cfg['Chalder_FS']['Answer']['9']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['10']['1'] = "Less than usual"
    ses_cfg['Chalder_FS']['Answer']['10']['2'] = "No more than usual"
    ses_cfg['Chalder_FS']['Answer']['10']['3'] = "More than usual"
    ses_cfg['Chalder_FS']['Answer']['10']['4'] = "Much more than usual"
    ses_cfg['Chalder_FS']['Answer']['11'] = {}
    ses_cfg['Chalder_FS']['Answer']['11']['1'] = "Better than usual"
    ses_cfg['Chalder_FS']['Answer']['11']['2'] = "No worse than usual"
    ses_cfg['Chalder_FS']['Answer']['11']['3'] = "Worse than usual"
    ses_cfg['Chalder_FS']['Answer']['11']['4'] = "Much worse than usual"

    ses_cfg['Environment'] = {}
    ses_cfg['Environment']['Description'] = "Experimental environment in which the session took place"
    ses_cfg['Environment']['Answer'] = {}
    ses_cfg['Environment']['Answer'][
        'Lab'] = "Laboratory, i.e., experimental room in the Centre of Neuroergonomie of ISAE-SUPAERO"
    ses_cfg['Environment']['Answer'][
        'Simulator'] = "Simulator, i.e., PEGASE flight simulator (Airbus A300 cockpit) of ISAE-SUPAERO"
    
    # Convert and write JSON object to file
    if not os.path.exists(bids_sessionpath):
        os.makedirs(bids_sessionpath)
    with open(op.join(bids_sessionpath,"sessions.json"), "w") as outfile:
        json.dump(ses_cfg, outfile,indent=3)
        
        
        
def create_run_config(bids_runpath):
    run_cfg = {}

    run_cfg['Run'] = {}
    run_cfg['Run']['Description'] = "Run (i.e., batch) number"

    run_cfg['Workload_Order'] = {}
    run_cfg['Workload_Order'][
        'Description'] = "Order of execution of the tree tasks of a run, i.e., 'Supervision', 'Easy' and 'Hard' "
    run_cfg['Notes'] = {}
    run_cfg['Notes'][
        'Description'] = "Notes regarding the data collection process and/or the recordings."

    run_cfg['KSS'] = {}
    run_cfg['KSS'][
        'Description'] = "Karolisnka Sleepyness Scale - The participant has to score each item regarding his/her level of sleepyness at the beginning of the run"
    run_cfg['KSS']['Answer'] = {}
    run_cfg['KSS']['Answer']['1'] = "Subject extremely alert"
    run_cfg['KSS']['Answer']['2'] = "Very alert"
    run_cfg['KSS']['Answer']['3'] = "Alert"
    run_cfg['KSS']['Answer']['4'] = "Fairly alert"
    run_cfg['KSS']['Answer']['5'] = "Neither alert or in sleep mode"
    run_cfg['KSS']['Answer']['6'] = "Few signs of sleepyness"
    run_cfg['KSS']['Answer']['7'] = "Sleepy, no effort to keep alert"
    run_cfg['KSS']['Answer']['8'] = "Sleepy, noticeable effort to keep alert"
    run_cfg['KSS']['Answer']['9'] = "Extremely sleepy, great effort to keep alert, fighting with sleepiness"

    run_cfg['Flickers'] = {}
    run_cfg['Flickers'][
        'Description'] = " Subjective evaluation of stimuli comfort - The participant has to score each item after the end of the run. The c-VEP flkickers are here referred as 'stimuli'"
    run_cfg['Flickers']['Question'] = {}
    run_cfg['Flickers']['Question'][
        '1'] = "Visual Comfort - Did you find the stimuli as visually fatiguing, neutral or comfortable?"
    run_cfg['Flickers']['Question']['2'] = "Mental Fatigue - Waht is your level of mental fatigue after the experiment?"
    run_cfg['Flickers']['Question']['3'] = "Distraction - Were the visual stimuli intrusive, neutral or discreet?"
    run_cfg['Flickers']['Answer'] = {}
    run_cfg['Flickers']['Answer']['1_range'] = "1 (Fatiguing) to 11 (Comfortable)"
    run_cfg['Flickers']['Answer']['2_range'] = "1 (Fatigued) to 11 (Not fatigued)"
    run_cfg['Flickers']['Answer']['3_range'] = "1 (Intrusive) to 11 (Discreet)"

    run_cfg['NASA_Easy'] = {}
    run_cfg['NASA_Easy'][
        'Description'] = "NASA-TLX - Hart and Saveland's NASA Task Load Index (TLX) method assesses work load within a 21 gradations scale. The participant has to score each item after the run, referred to the 'Easy' task"
    run_cfg['NASA_Easy']['Question'] = {}
    run_cfg['NASA_Easy']['Question']['1'] = "Mental Demand - How mentally demanding was the task?"
    run_cfg['NASA_Easy']['Question']['2'] = "Physical Demand - How physically demanding was the task?"
    run_cfg['NASA_Easy']['Question']['3'] = "Temporal Demand - How hurried or rushed was the pace of the task?"
    run_cfg['NASA_Easy']['Question'][
        '4'] = "Performance - How successful were you in accomplishing what you were asked to do?"
    run_cfg['NASA_Easy']['Question'][
        '5'] = "Effort - How hard did you have to work to accomplish your level of performance?"
    run_cfg['NASA_Easy']['Question'][
        '6'] = "Frustrtion - How insecure, discouraged, irritated, stressed and annoyed were you?"
    run_cfg['NASA_Easy']['Answer'] = {}
    run_cfg['NASA_Easy']['Answer']['1_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Easy']['Answer']['2_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Easy']['Answer']['3_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Easy']['Answer']['4_range'] = "1 (Perfect) to 21 (Failure)"
    run_cfg['NASA_Easy']['Answer']['5_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Easy']['Answer']['6_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Easy']['Answer']['7_range'] = "1 (Very Low) to 21 (Very High)"

    run_cfg['NASA_Hard'] = {}
    run_cfg['NASA_Hard'][
        'Description'] = "NASA-TLX - Hart and Saveland's NASA Task Load Index (TLX) method assesses work load within a 21 gradations scale. The participant has to score each item after the run, referred to the 'Hard' task"
    run_cfg['NASA_Hard']['Question'] = {}
    run_cfg['NASA_Hard']['Question']['1'] = "Mental Demand - How mentally demanding was the task?"
    run_cfg['NASA_Hard']['Question']['2'] = "Physical Demand - How physically demanding was the task?"
    run_cfg['NASA_Hard']['Question']['3'] = "Temporal Demand - How hurried or rushed was the pace of the task?"
    run_cfg['NASA_Hard']['Question'][
        '4'] = "Performance - How successful were you in accomplishing what you were asked to do?"
    run_cfg['NASA_Hard']['Question'][
        '5'] = "Effort - How hard did you have to work to accomplish your level of performance?"
    run_cfg['NASA_Hard']['Question'][
        '6'] = "Frustrtion - How insecure, discouraged, irritated, stressed and annoyed were you?"
    run_cfg['NASA_Hard']['Answer'] = {}
    run_cfg['NASA_Hard']['Answer']['1_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Hard']['Answer']['2_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Hard']['Answer']['3_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Hard']['Answer']['4_range'] = "1 (Perfect) to 21 (Failure)"
    run_cfg['NASA_Hard']['Answer']['5_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Hard']['Answer']['6_range'] = "1 (Very Low) to 21 (Very High)"
    run_cfg['NASA_Hard']['Answer']['7_range'] = "1 (Very Low) to 21 (Very High)"

    run_cfg['Performance_Easy'] = {}
    run_cfg['Performance_Easy'][
        'Description'] = "Performance of the subject in the Easy task of the run, based on three indexes."
    run_cfg['Performance_Easy']['Mean Reaction Time'] = {}
    run_cfg['Performance_Easy']['Mean Reaction Time'][
        'Description'] = "Mean time passed between the occurrence of an event on the Monitoring subtask and the response of the subject with the correct key"
    run_cfg['Performance_Easy']['Mean Reaction Time']['Units'] = "Seconds"
    run_cfg['Performance_Easy']['Tracking Accuracy'] = {}
    run_cfg['Performance_Easy']['Tracking Accuracy'][
        'Description'] = "If 'Lab' environment: Score in [0, 1] inversely proportional to the total RMSE between the Tracking subtask cursor position and the center of the subtask (goal position)." \
                         "If 'Simulator' score in [0, 1] proportional to the closeness of the altitude and heading trajectories to those of an ideal pilot during the task."
    run_cfg['Performance_Easy']['Tracking Accuracy']['Range'] = "0 to 1"
    run_cfg['Performance_Easy']['Communication Accuracy'] = {}
    run_cfg['Performance_Easy']['Communication Accuracy'][
        'Description'] = "Number of correctly responded Communication subtask calls over the total calls."
    run_cfg['Performance_Easy']['Communication Accuracy']['Range'] = "0 to 1"

    run_cfg['Performance_Hard'] = {}
    run_cfg['Performance_Hard'][
        'Description'] = "Performance of the subject in the Hard task of the run, based on three indexes."
    run_cfg['Performance_Hard']['Mean Reaction Time'] = {}
    run_cfg['Performance_Hard']['Mean Reaction Time'][
        'Description'] = "Mean time passed between the occurrence of an event on the Monitoring subtask and the response of the subject with the correct key"
    run_cfg['Performance_Hard']['Mean Reaction Time']['Units'] = "Seconds"
    run_cfg['Performance_Hard']['Tracking Accuracy'] = {}
    run_cfg['Performance_Hard']['Tracking Accuracy'][
        'Description'] = "If 'Lab' environment: Score in [0, 1] inversely proportional to the total RMSE between the Tracking subtask cursor position and the center of the subtask (goal position)." \
                         "If 'Simulator' score in [0, 1] proportional to the closeness of the altitude and heading trajectories to those of an ideal pilot during the task."
    run_cfg['Performance_Hard']['Tracking Accuracy']['Range'] = "0 to 1"
    run_cfg['Performance_Hard']['Communication Accuracy'] = {}
    run_cfg['Performance_Hard']['Communication Accuracy'][
        'Description'] = "Number of correctly responded Communication subtask calls over the total calls."
    run_cfg['Performance_Hard']['Communication Accuracy']['Range'] = "0 to 1"

    # Convert and write JSON object to file
    if not os.path.exists(bids_runpath):
        os.makedirs(bids_runpath)
    with open(op.join(bids_runpath,"runs.json"), "w") as outfile:
        json.dump(run_cfg, outfile,indent=3)


# //////////////////////////////////////////////////
# def main(
path = "D:/s.velut/Documents/Thèse/Protheus_PHD/Data/PROTEUS/rec/"
path_behavioral = 'D:/s.velut/Documents/Thèse/Protheus_PHD/Data/PROTEUS/Proteus_data2.csv'
bids_root = "D:/s.velut/Documents/Thèse/Protheus_PHD/Data/PROTEUS/BIDS/"
# participants = ['s2', 's4', 's5', 's6', 's7', 's8', 's10', 's11', 's12', 's13', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's24', 's25']
participants = ['s2', 's4', 's5', 's6']
sessions = [str(s) for s in list(range(1,7))]
runs = ["1", "2"]
# participants = ["s2"]
# sessions = ["2"]
# runs = ["1"]
task = "matb"
simu = True
# //////////////////////////////////////////////////


df_behavioral = pd.read_csv(path_behavioral, header=0)
fields_subj = ['Participant', 'Age', 'Sex', 'Laterality', 'Education', 'VideogameQ1', 'VideogameQ2', 'VideogameQ3', 'VideogameQ4', 'Flight_experience', 'DES']
fields_session = ['Session', 'Date', 'Hour', 'VASF', 'ChalderFS', 'Environment']
fields_run = ['Batch', 'KSS(before)', 'Flickers', 'NASA_Easy', 'NASA_Hard', 'ChalderFS', 'Workload_Order', 'Notes', 'Performance_Easy', 'Performance_Hard']


true_channels = ['Fp1','Fz','F3','F7','F9','FC5','FC1','C3','T7','P9','CP5','CP1','Pz','P3','P7','O1','Oz','O2','P4','P8','P10','CP6','CP2','Cz','C4','T8','FT8','FC6','FC2','F4','F8','Fp2']

for participant in participants:
    for session in sessions:
        for run in runs:
            for task in ["calib", "matb"]:

                df_line = df_behavioral.loc[(df_behavioral['Participant'] == int(participant[1:])) & (
                            df_behavioral['Session'] == int(session)) & (df_behavioral['Batch'] == int(run))]
                if list(df_line['Environment'])[0]:
                    if list(df_line['Environment'])[0] == 'Lab':
                        simu = False
                    elif list(df_line['Environment'])[0] == 'Lab':
                        simu = True

                simuprefix = "simu" if simu else 'lab'


                if os.path.isfile(op.join(path + participant + "/" + simuprefix + "/",
                                          '_'.join([participant, session, run, task + ".xdf"]))):

                    #### GET DATA
                    streams, header = pyxdf.load_xdf(op.join(path+participant+"/"+simuprefix+"/", '_'.join([participant,session,run,task+".xdf"])))
                    keys = {''.join([j for j in streams[i]['info']['name'][0] if not j.isdigit() and j!="-"]):i for i in range(len(streams))}

                    ############# BEHAVIORAL DATA #############

                    create_general_config(bids_root)

                    df_subj = df_behavioral.loc[(df_behavioral['Participant'] == int(participant[1:]))][fields_subj].dropna()
                    df_session = df_behavioral.loc[(df_behavioral['Participant'] == int(participant[1:])) & (df_behavioral['Session'] == int(session))][fields_session].dropna()
                    df_run = df_line[fields_run]
                    df_run.columns = ['Run', 'KSS', 'Flickers', 'NASA_Easy', 'NASA_Hard', 'ChalderFS', 'Workload_Order', 'Notes', 'Performance_Easy', 'Performance_Hard']

                    ### Save demographic and behavioral data

                    if not os.path.isfile(op.join(bids_root, 'demographics.json')):
                        create_subj_config(bids_root)
                    if os.path.isfile(op.join(bids_root, 'demographics.tsv')):
                        df_subj_previous = pd.read_csv(op.join(bids_root, 'demographics.tsv'), sep='\t')
                        if df_subj['Participant'].values[0] not in df_subj_previous['Participant'].values:
                            df_subj_updated = pd.concat([df_subj_previous, df_subj]).sort_values(by=['Participant'])
                            df_subj_updated.to_csv(op.join(bids_root, 'demographics.tsv'), sep='\t', index=False)
                    else:
                        df_subj.to_csv(op.join(bids_root, 'demographics.tsv'), sep='\t', index=False)

                    ses_path = op.join(bids_root, "sub-" + participant)
                    if not os.path.isfile(op.join(ses_path, 'sessions.json')):
                        create_session_config(ses_path)
                    if os.path.isfile(op.join(ses_path, 'sessions.tsv')):
                        df_session_previous = pd.read_csv(op.join(ses_path, 'sessions.tsv'), sep='\t')
                        if df_session['Session'].values[0] not in df_session_previous['Session'].values:
                            df_session_updated = pd.concat([df_session_previous, df_session]).sort_values(by=['Session'])
                            df_session_updated.to_csv(op.join(ses_path, 'sessions.tsv'), sep='\t', index=False)
                    else:
                        df_session.to_csv(op.join(ses_path, 'sessions.tsv'), sep='\t', index=False)


                    run_path = op.join(bids_root, "sub-" + participant, "ses-" + session)
                    if not os.path.isfile(op.join(run_path, 'runs.json')):
                        create_run_config(run_path)
                    if os.path.isfile(op.join(run_path, 'runs.tsv')):
                        df_run_previous = pd.read_csv(op.join(run_path, 'runs.tsv'), sep='\t')
                        if df_run['Run'].values[0] not in df_run_previous['Run'].values:
                            df_run_updated = pd.concat([df_run_previous, df_run], ignore_index=True).sort_values(by=['Run'])
                            df_run_updated.to_csv(op.join(run_path, 'runs.tsv'), sep='\t', index=False)
                    else:
                        df_run.to_csv(op.join(run_path, 'runs.tsv'), sep='\t', index=False)







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
                    raw.rename_channels(dict(zip(raw.ch_names, true_channels)))
                    montage = mne.channels.read_custom_montage(path + "channels_proteus32_final.xyz", coord_frame='head')
                    raw.set_montage(montage)

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
                        nb_point = nb_point - nb_point % 3
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
                        onset_frame = min([int((onset-eeg_start)*sfreq), stim_chan.size-1])
                        stim_chan[0, onset_frame] = label
                        stim_onset[i] = (onset-eeg_start)
                        stim_label.append(label)
                        i+=1

                    if task=='calib':
                        i=0
                        for onset, label in zip(code_time, code):
                            onset_frame = min([int((onset - eeg_start) * sfreq), stim_chan.size-1])
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
                        if 'PEGASE_Data' in keys:
                            fs_stream = streams[keys['PEGASE_Data']]
                            fs_info = fs_stream['info']
                            fs_data = fs_stream['time_series']
                            fs_time = fs_stream['time_stamps']

                            ### create metadata
                            metadata_pegase = pd.read_csv(path + "pegase_metadata.csv", sep=';')

                            ### Channel in the flight pegase data json
                            fs_cfg = {}
                            fs_cfg["SamplingFrequency"] = fs_info['effective_srate']
                            fs_cfg["NominalSamplingFrequency"] = fs_info['nominal_srate']
                            fs_cfg['StartTime'] = fs_time[0]
                            fs_cfg["BIDSVersion"] = 'n/a'
                            fs_cfg['TaskName'] = task
                            fs_cfg[
                                'TaskDescription'] = "Calibration before main experiment" if task == "calib" else "Main experiment by performing MATB task"
                            fs_cfg[
                                'Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task == "calib" else "Perform the 3 task of the MATB"
                            fs_cfg['Manufacturer'] = "ISAE-SUPAERO"  ## A changer
                            fs_cfg['ManufacturerModelName'] = "PEGASE"  ## A changer
                            fs_cfg['DeviceSerialNumber'] = "None"  ## A changer
                            fs_cfg['SoftwareVersions'] = "PEGASE"  ## A changer
                            fs_cfg['InstitutionName'] = 'ISAE-SUPAERO'
                            fs_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
                            fs_cfg['InstitutionalDepartmentName'] = "CNE"
                            fs_cfg = get_FlightSimuChannels(fs_cfg, metadata_pegase)

                            ### Create tsv data
                            fs_header = metadata_pegase['Nom'].values
                            fs_pd = pd.DataFrame(fs_data, columns=fs_header)

                            ### save files
                            fs_path = op.join(bids_root, "sub-" + participant, "ses-" + session, "fsimu")
                            fs_data_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_flightsimulatordata.tsv"
                            fs_cfg_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_flightsimulatordata.json"
                            if not os.path.exists(fs_path):
                                os.makedirs(fs_path)
                            fs_pd.to_csv(op.join(fs_path, fs_data_name), sep='\t')
                            # Convert and write JSON object to file
                            with open(op.join(fs_path, fs_cfg_name), "w") as outfile:
                                json.dump(fs_cfg, outfile, indent=3)


                        if 'Flying_metrics' in keys:

                            fsm_stream = streams[keys['Flying_metrics']]
                            fsm_info = fsm_stream['info']
                            fsm_data = fsm_stream['time_series']
                            fsm_time = fsm_stream['time_stamps']


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


                            fsm_header = ["iWaypoint", "Dist", "Heading", "Perf", "Altitude_visee", "Vitesse"]
                            fsm_pd = pd.DataFrame(fsm_data,columns=fsm_header)



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

                        if 'Focus' in keys:
                            foc_stream = streams[keys['Focus']]
                            foc_info = foc_stream['info']
                            foc_data = foc_stream['time_series']
                            foc_time = foc_stream['time_stamps']

                            foc_cfg = {}
                            foc_cfg["SamplingFrequency"] = foc_info['effective_srate']
                            foc_cfg["NominalSamplingFrequency"] = foc_info['nominal_srate']
                            foc_cfg['StartTime'] = foc_time[0]
                            foc_cfg["BIDSVersion"] = 'n/a'
                            foc_cfg['TaskName'] = task
                            foc_cfg[
                                'TaskDescription'] = "Calibration before main experiment" if task == "calib" else "Main experiment by performing MATB task"
                            foc_cfg[
                                'Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task == "calib" else "Perform the 3 task of the MATB"
                            foc_cfg['Manufacturer'] = "ISAE-SUPAERO"  ## A changer
                            foc_cfg['ManufacturerModelName'] = "NONE"  ## A changer
                            foc_cfg['DeviceSerialNumber'] = "None"  ## A changer
                            foc_cfg['SoftwareVersions'] = "NONE"  ## A changer
                            foc_cfg['InstitutionName'] = 'ISAE-SUPAERO'
                            foc_cfg['InstitutionAddress'] = "10 Avenue Avenue Marc Pélegrin, BP 54032, 31055 Toulouse Cedex 4 France"
                            foc_cfg['InstitutionalDepartmentName'] = "CNE"
                            foc_cfg['DataDescription'] = 'Prediction of the focus of the participant after accumulation'

                            foc_pd = pd.DataFrame(foc_data, columns=["focus"])

                            ### path to save Focus data
                            foc_path = op.join(bids_root, "sub-" + participant, "ses-" + session, "extradata")
                            foc_data_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_focusdata.tsv"
                            foc_cfg_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_focusdata.json"
                            if not os.path.exists(foc_path):
                                os.makedirs(foc_path)
                            foc_pd.to_csv(op.join(foc_path, foc_data_name), sep='\t')
                            # Convert and write JSON object to file
                            with open(op.join(foc_path, foc_cfg_name), "w") as outfile:
                                json.dump(foc_cfg, outfile, indent=3)


                        if 'MATB' in keys:
                            matb_stream = streams[keys['MATB']]
                            matb_info = matb_stream['info']
                            matb_data = matb_stream['time_series']
                            matb_time = matb_stream['time_stamps']

                            matb_cfg = {}
                            matb_cfg["SamplingFrequency"] = matb_info['effective_srate']
                            matb_cfg["NominalSamplingFrequency"] = matb_info['nominal_srate']
                            matb_cfg['StartTime'] = matb_time[0]
                            matb_cfg["BIDSVersion"] = 'n/a'
                            matb_cfg['TaskName'] = task
                            matb_cfg[
                                'TaskDescription'] = "Calibration before main experiment" if task == "calib" else "Main experiment by performing MATB task"
                            matb_cfg[
                                'Instructions'] = "Look at the cued Flicker for 10seconds after the cue" if task == "calib" else "Perform the 3 task of the MATB"
                            matb_cfg['Manufacturer'] = "ISAE-SUPAERO"  ## A changer
                            matb_cfg['ManufacturerModelName'] = "NONE"  ## A changer
                            matb_cfg['DeviceSerialNumber'] = "None"  ## A changer
                            matb_cfg['SoftwareVersions'] = "NONE"  ## A changer
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

                            matb_header = ['time', 'label', 'value']
                            matb_pd = pd.DataFrame(matb_data, columns=matb_header)

                            ### path to save MATB data
                            matb_path = op.join(bids_root, "sub-" + participant, "ses-" + session, "extradata")
                            matb_data_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_matbdata.tsv"
                            matb_cfg_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_matbdata.json"
                            if not os.path.exists(matb_path):
                                os.makedirs(matb_path)
                            matb_pd.to_csv(op.join(matb_path, matb_data_name), sep='\t')
                            # Convert and write JSON object to file
                            with open(op.join(matb_path, matb_cfg_name), "w") as outfile:
                                json.dump(matb_cfg, outfile, indent=3)


                        if 'Prediction' in keys:
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


                            ### create tsv
                            pred_header = ['rate','onset','cIndex','cBits','results']
                            n_values = len(pred_data)
                            dict_list = [eval(pred_data[i][0].replace('Timestamp','')) for i in range(n_values)]
                            pred_data = [[dict_list[i]['rate'], dict_list[i]['epoch']['onset'],dict_list[i]['epoch']['context']['index'],dict_list[i]['epoch']['context']['bits'],eval(pred_data[i][1])['result']] for i in range(n_values)]
                            pred_pd = pd.DataFrame(pred_data,columns=pred_header)

                            ### save file
                            pred_path = op.join(bids_root, "sub-" + participant, "ses-" + session, "extradata")
                            pred_data_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_predictiondata.tsv"
                            pred_cfg_name = "sub-" + participant + "_ses-" + session + "_task-" + task + simuprefix + "_run-" + run + "_predictiondata.json"
                            if not os.path.exists(pred_path):
                                os.makedirs(pred_path)
                            pred_pd.to_csv(op.join(pred_path, pred_data_name), sep='\t')
                            # Convert and write JSON object to file
                            with open(op.join(pred_path, pred_cfg_name), "w") as outfile:
                                json.dump(pred_cfg, outfile, indent=3)








# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", help="Path to the data to tranform")
#     parser.add_argument("--bids_root", help="Patht to folder being in BIDS format")
#     parser.add_argument("--participant",help="Participant id (ex: s12) which data we want to transform in BIDS")
#     parser.add_argument("--session",default="1",help="session of the data to transform in BIDS")
#     parser.add_argument("--run",default="1",help="run of the data to transform in BIDS")
#     parser.add_argument("--task",help="task of the data to transform in BIDS (only 'calib' or 'matb')")
#     parser.add_argument("--simu",help="Boolean to know if it was in the simulator or not")
#     parser.add_argument("--general_config",default='False',help="Boolean to know if you want to create the general config or not")
#
#     args = parser.parse_args()
#     general_config = eval(args.general_config)
#     print(general_config)
#     if general_config:
#         create_general_config(args.bids_root)
#     else:
#         main(args.path,args.bids_root,args.participant,args.session,args.run,args.task,args.simu)
