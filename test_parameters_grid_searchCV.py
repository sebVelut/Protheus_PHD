import pandas as pd
from EEG2CodeKeras import (basearchi,
                           basearchitest_batchnorm,
                           basearchi_patchembedding,
                           basearchi_patchembeddingdilation,
                           trueVanilliaEEG2Code,
                           vanilliaEEG2Code,
                           vanilliaEEG2Code2,
                           EEGnet_Inception)
from _utils import make_preds_accumul_aggresive, make_preds_pvalue

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tensorflow import keras
import mne
from mne.decoding import Vectorizer, CSP
import sys
import os
import numpy as np
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40
np.set_printoptions(threshold=sys.maxsize)


# general parameters
fps = 60
sfreq = 500
nb_point_per_sec = 60
n_class=4
fps = 60
window_size = 0.25
paradigm_type = "burst100"
n_cal = 7

# Check every participant
## Initializing an array to store the score of every participant with the different classifier
participant_score = np.zeros((12,5))
for id_parti in range(1,13):
    # Get the data of the participant id_parti
    print("Getting the data of participant :",id_parti)
    
    if paradigm_type=="mseq40":
        dataset_moabb = CasitllosCVEP40()
    elif paradigm_type=="mseq100":
        dataset_moabb = CasitllosCVEP100()
    elif paradigm_type=="burst40":
        dataset_moabb = CasitllosBurstVEP40()
    elif paradigm_type=="burst100":
        dataset_moabb = CasitllosBurstVEP100()
    else:
        print("ArgumentError : the type of paradigm doesn't exist. Check it !")
        sys.exit()
    
    dataset_moabb = CasitllosBurstVEP100()
    # print(paradigm.get_data(dataset_moabb,subjects=[1,2]))
    raw = dataset_moabb.get_data(subjects=[id_parti])[id_parti]["0"]["0"]


    """"" Create code label"""
    # print(raw.ch_names)
    temp_raw = raw.copy()
    trial_chan = temp_raw.pick_channels(["stim_trial"],verbose=False)
    data = trial_chan.get_data()[0]
    labels_code = np.array(list(filter(lambda num: num != 0, data)),dtype=int)-200
    raw = raw.filter(l_freq=1, h_freq=20, method="fir", verbose=True)
    mne.set_eeg_reference(raw, 'average', copy=False, verbose=False)
    # n_channels = len(raw.ch_names)

    events = mne.find_events(raw,["stim_epoch"]) #mne.events_from_annotations(raw, event_id='auto', verbose=False)
    labels = events[..., -1]
    print(events.shape)
    n_samples_windows = int(window_size*sfreq)
    n_trial_per_class = int(len(raw)/n_class)


    n_cal = 7
    n_cal_min = 1
    n_cal_max = 10
    accuracy = []


    # Separate the data in train and tests datasets
    epochs = mne.Epochs(raw,events,{"0":100,"1":101},picks=raw.ch_names[:-2],tmin=-0.01,tmax=0.3)
    X=epochs.get_data()
    print(n_class*n_cal*nb_point_per_sec)
    label = epochs.events[...,-1]-100
    size_train = int((2.2-window_size)*n_class*n_cal*nb_point_per_sec)
    print(size_train)
    X_train = X[:size_train]
    X_test = X[size_train:]
    print(X_test.shape)
    # print("test data",X_test)
    Y_train = label[:size_train]-100
    Y_test = label[size_train:]-100
    labels_test = labels_code[n_class*n_cal:]

    X_std = X_train.std(axis=0)
    X_train /= X_std + 1e-8
    X_std = X_test.std(axis=0)
    X_test /= X_std + 1e-8

    txt = "sansBalancing"
    #balancing the number of ones and zeros
    # txt = "avecBalancing"
    # print("balancing the number of ones and zeros\n")
    # rus = RandomUnderSampler()
    # counter=np.array(range(0,len(Y_train))).reshape(-1,1)
    # index,_ = rus.fit_resample(counter,Y_train[:])
    # X_train = np.squeeze(X_train[index,:,:], axis=1)
    # Y_train = np.squeeze(Y_train[index])

    # Creating the different classifier
    print("Creating the different pipelines")
    RG_MDM = make_pipeline(XdawnCovariances(nfilter=6, estimator="lwf", xdawn_estimator="lwf"),MDM())
    RG_LDA = make_pipeline(XdawnCovariances(nfilter=6, estimator="lwf", xdawn_estimator="lwf"),
        TangentSpace(),
        LDA(solver="lsqr", shrinkage="auto"))
    RG_SVC = make_pipeline(XdawnCovariances(nfilter=6, estimator="lwf", xdawn_estimator="lwf"),
        TangentSpace(),
        svm.SVC())
    Xd_LDA = make_pipeline(Xdawn(nfilter=6, estimator="lwf"),Vectorizer(),
        LDA(solver="lsqr", shrinkage="auto"))
    CSP_LDA = make_pipeline(CSP(n_components=4, reg=None, log=True, norm_trace=False),LDA())

    print("Creating the different parameters to check\n")
    param_RG_MDM = {'xdawncovariances__nfilter':[4,6,8,10],
                    'xdawncovariances__estimator':['lwf','oas'],
                    'xdawncovariances__xdawn_estimator':['lwf','oas']}
    param_RG_LDA = {'xdawncovariances__nfilter':[4,6,8,10],
                    'xdawncovariances__estimator':['lwf','oas'],
                    'xdawncovariances__xdawn_estimator':['lwf','oas'],
                    "lineardiscriminantanalysis__solver":["lsqr"]}
    param_RG_SVC = {'xdawncovariances__nfilter':[4,6,8,10],
                    'xdawncovariances__estimator':['lwf','oas'],
                    'xdawncovariances__xdawn_estimator':['lwf','oas'],
                    "svc__kernel":["rbf","poly"],
                    "svc__C":[0.5,1,5,10]}
    param_Xd_LDA = {'xdawn__nfilter':[4,6,8,10],
                    'xdawn__estimator':['lwf','oas'],
                    "lineardiscriminantanalysis__solver":["lsqr"]}
    
    print("Creating the different gridsearchCV\n")
    clf_RG_MDM = GridSearchCV(RG_MDM,param_RG_MDM)
    clf_RG_LDA = GridSearchCV(RG_LDA,param_RG_LDA)
    clf_RG_SVC = GridSearchCV(RG_SVC,param_RG_SVC)
    clf_Xd_LDA = GridSearchCV(Xd_LDA,param_Xd_LDA)

    # x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, shuffle=True)
    print("Fitting the RG+MDM\n")
    clf_RG_MDM.fit(X_train,Y_train)
    print(clf_RG_MDM.best_estimator_)
    print("Fitting the RG+LDA\n")
    clf_RG_LDA.fit(X_train,Y_train)
    print(clf_RG_LDA.best_estimator_)
    print("Fitting the RG+SVC\n")
    clf_RG_SVC.fit(X_train,Y_train)
    print(clf_RG_SVC.best_estimator_)
    print("Fitting the Xdawn+LDA\n")
    clf_Xd_LDA.fit(X_train,Y_train)
    print(clf_Xd_LDA.best_estimator_)

    participant_score[id_parti,0] = balanced_accuracy_score(Y_test,clf_RG_MDM.predict(X_test))
    participant_score[id_parti,1] = balanced_accuracy_score(Y_test,clf_RG_LDA.predict(X_test))
    participant_score[id_parti,2] = balanced_accuracy_score(Y_test,clf_RG_SVC.predict(X_test))
    participant_score[id_parti,3] = balanced_accuracy_score(Y_test,clf_Xd_LDA.predict(X_test))

    keras.backend.clear_session()

df = pd.DataFrame({"Xdawn+RG+MDM":clf_RG_MDM.best_estimator_,"Xdawn+RG+LDA":clf_RG_LDA.best_estimator_,
                   "Xdawn+RG+SVC":clf_RG_SVC.best_estimator_,"Xdawn+LDA":clf_Xd_LDA.best_estimator_})
df.to_csv("C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\results\\gridSearchCV_score_{0}.csv".format(txt))
