import h5py
import pandas as pd
from Scripts.EEG2CodeKeras import (basearchi,
                           basearchitest_batchnorm,
                           basearchi_patchembedding,
                           basearchi_patchembeddingdilation,
                           trueVanilliaEEG2Code,
                           vanilliaEEG2Code,
                           vanilliaEEG2Code2,
                           EEGnet_Inception)
from Scripts._utils import make_preds_accumul_aggresive, make_preds_pvalue


from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tensorflow import keras
import mne
from mne.decoding import Vectorizer, CSP
import os
import sys
import seaborn as sns

from moabb.datasets import download as dl
from moabb.evaluations.evaluations import WithinSessionEvaluation
from moabb.paradigms import CVEP
from moabb.datasets import Cattan2019_VR
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40


import sys
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

fps = 60
sfreq = 500


n_class=4
fps = 60
window_size = 0.25

id_min = 1
id_max = 13
nb_point_per_sec = 60
n_cal = 7
results_df = pd.DataFrame(columns=["Score","time","subject","nb_cal","clf","score_01"])

dataset_moabb = CasitllosBurstVEP100()
paradigm = CVEP()
print(paradigm.n_classes)
# print(paradigm.get_data(dataset_moabb,subjects=[1,2]))
# raw = dataset_moabb.get_data(subjects=[id_parti])[id_parti]["0"]["0"]

""""" Create code label"""
 
pipelines = {}
pipelines["RG+LDA"]=make_pipeline(
    XdawnCovariances(
        nfilter=6, estimator="oas", xdawn_estimator="lwf"
    ),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto"),
)
# pipelines["RG+SVC"] = make_pipeline(XdawnCovariances(nfilter=6, estimator="oas", xdawn_estimator="lwf"),
# TangentSpace(),
# svm.SVC())
# pipelines["Xd+LDA"] = make_pipeline(Xdawn(nfilter=6, estimator="lwf"),Vectorizer(),LDA(solver="lsqr", shrinkage="auto"))

# # # paradigm = CVEP(resample=128)
# print("charging dataset")
# # print(dataset_moabb.event_id)
# dataset_moabb.subject_list = dataset_moabb.subject_list[0:3]
# # print(dataset_moabb.subject_list)
# datasets = [dataset_moabb]
# overwrite = True  # set to True if we want to overwrite cached results
# evaluation = WithinSessionEvaluation(
#     paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite,hdf5_path="./results",save_model=True
# )

# results = evaluation.process(pipelines)
# print(results)

with open(
    "C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\results\\results\\CVEP\\WithinSessionEvaluation\\results_examples.hdf5",
    "rb",
) as pickle_file:
    f = h5py.File(pickle_file)
    print(np.fromfile(pickle_file, dtype=float))

    print("classif",f[list(f.keys())[0]][()])


# dataset_moabb = CasitllosBurstVEP100()
# raw = dataset_moabb.get_data(subjects=[1])[1]["0"]["0"]
# temp_raw = raw.copy()
# trial_chan = temp_raw.pick_channels(["stim_trial"],verbose=False)
# data = trial_chan.get_data()[0]
# labels_code = np.array(list(filter(lambda num: num != 0, data)),dtype=int)-200

# events = mne.find_events(raw,["stim_epoch"]) #mne.events_from_annotations(raw, event_id='auto', verbose=False)
# print("events",events)
# labels = events[..., -1]
# print(events.shape)
# n_samples_windows = int(window_size*sfreq)
# n_trial_per_class = int(len(raw)/n_class)

# epochs = mne.Epochs(raw,events,{"0":100,"1":101},picks=raw.ch_names[:-2],tmin=-0.01,tmax=0.3)
# X=epochs.get_data()
# label = epochs.events[...,-1]-100

# X_std = X.std(axis=0)
# X /= X_std + 1e-8

# y_pred = clf.predict(X)
# print(balanced_accuracy_score(label,y_pred))
# labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
#     clf.predict(X), dataset_moabb.codes, min_len=30, sfreq=nb_point_per_sec, consecutive=50, window_size=window_size
# )
# accuracy1 = np.round(accuracy_score(labels_code[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
# print(accuracy1)