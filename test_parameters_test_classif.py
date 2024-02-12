import pandas as pd
from sklearn.cross_decomposition import CCA
from EEG2CodeKeras import (basearchi,
                           basearchitest_batchnorm,
                           basearchi_patchembedding,
                           basearchi_patchembeddingdilation,
                           trueVanilliaEEG2Code,
                           vanilliaEEG2Code,
                           vanilliaEEG2Code2,
                           EEGnet_Inception)
from Scripts.SPDNet.optimizer.riemannian_adam import RiemannianAdam
from _utils import make_preds_accumul_aggresive, make_preds_pvalue
import time


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
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts\\SPDNet")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40
from Scripts.SPDNet.spd_net_2_tensorflow import SPDNet_AJD
from Scripts.SPDNet.spd_net_tensorflow import SPDNet_Tensorflow


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
id_max = 2
nb_point_per_sec = 60
results_df = pd.DataFrame(columns=["Score","time","subject","nb_cal","clf","score_01","time_training"])

for id_parti in range(id_min,id_max):
    print("Participant n°:",id_parti)
    participant = 'P'+str(id_parti)
    # path = '/'.join(['C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Class4', participant])

    # file_name = '_'.join([participant, 'mseq40.set'])
    # file_name = '_'.join([participant, 'mseq100.set'])
    # file_name = '_'.join([participant, 'burst40.set'])
    # file_name = '_'.join([participant, 'burst100.set'])
    # file_name = '/'.join([path,  participant+'_whitemseq.set'])
    # file_name = '_'.join([participant, 'burst', 'oi_1.set'])
    ##########################
    # #### Test de récupérer les données avec moabb et le datasets créé

    dataset_moabb = CasitllosBurstVEP100()
    paradigm = CVEP()
    print(paradigm.n_classes)
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
    # raw.plot(block=True)
    # print(raw)
    # print(mne.events_from_annotations(raw, event_id='auto', verbose=False)[1])
    # print(dataset_moabb.paradigm)
    # print(paradigm.datasets)

    # print("fin\n\n\n")
    # assert(1==0)


    pipelines = {}
    pipelines["RG+LDA"]=make_pipeline(
        XdawnCovariances(
            nfilter=6, estimator="oas", xdawn_estimator="lwf"
        ),
        TangentSpace(),
        LDA(solver="lsqr", shrinkage="auto"),
    )
    pipelines["RG+SVC"] = make_pipeline(XdawnCovariances(nfilter=6, estimator="oas", xdawn_estimator="lwf"),
    TangentSpace(),
    svm.SVC())
    pipelines["Xd+LDA"] = make_pipeline(Xdawn(nfilter=6, estimator="lwf"),Vectorizer(),LDA(solver="lsqr", shrinkage="auto"))


    # paradigm = CVEP(resample=128)
    print("charging dataset")
    print(dataset_moabb.event_id)
    dataset_moabb.subject_list = dataset_moabb.subject_list[:]
    print(dataset_moabb.subject_list)
    datasets = [dataset_moabb]
    overwrite = True  # set to True if we want to overwrite cached results
    evaluation = WithinSessionEvaluation(
        paradigm=paradigm, datasets=datasets, suffix="examples", overwrite=overwrite
    )

    results = evaluation.process(pipelines)

    print("aaaaaaaaaaaaaaaaaaaaaaaa\n\n\n\n\n\n",results)

    fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

    sns.stripplot(
        data=results,
        y="score",
        x="pipeline",
        ax=ax,
        jitter=True,
        alpha=0.5,
        zorder=1,
        palette="Set1",
    )
    sns.pointplot(data=results, y="score", x="pipeline", ax=ax, palette="Set1")

    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1)

    plt.show()
    break

    ##########################
    # raw = mne.io.read_raw_eeglab(os.path.join(path, file_name), preload=True, verbose=False)

    # print(raw.ch_names)
    # to_drop = ["P9", "P10", "TP9", "TP10", "10", "21"]
    # raw = raw.drop_channels([ch for ch in raw.ch_names if ch in to_drop])
    # # raw = raw.drop_channels(["10", "21"])
    # keep = ["O1", "O2", "Oz", "P7", "P3", "P4", "P8", "Pz"]
    # keep = ["16", "18", "17", "15", "14", "19", "20", "13"] # electrodes to keep
    # raw = raw.drop_channels([i for i in raw.ch_names if i not in keep])

    # raw = raw.filter(l_freq=50.1, h_freq=49.9, method="iir", verbose=True)
    # raw = raw.filter(l_freq=1, h_freq=20, method="fir", verbose=True)
    # # raw.resample(480, npad='auto')
    # # Average re-referencing
    # mne.set_eeg_reference(raw, 'average', copy=False, verbose=False)
    # #raw = raw.filter(l_freq=5, h_freq=45, method="fir", verbose=True)
    # n_channels = len(raw.ch_names)
    # print("Channels :", n_channels)

    # # Strip the annotations that were script to make them easier to process
    # events, event_id = mne.events_from_annotations(raw, event_id='auto', verbose=False)
    # to_remove = []
    # for idx in range(len(raw.annotations.description)):
    #     if (('collects' in raw.annotations.description[idx]) or
    #         ('iti' in raw.annotations.description[idx]) or
    #         (raw.annotations.description[idx] == '[]')):
    #         to_remove.append(idx)
    #     else:
    #         code = raw.annotations.description[idx].split('_')[0]
    #         lab = raw.annotations.description[idx].split('_')[1]
    #         code = code.replace('\n', '')
    #         code = code.replace('[', '')
    #         code = code.replace(']', '')
    #         code = code.replace(' ', '')
    #         raw.annotations.description[idx] = code + '_' + lab

    # to_remove = np.array(to_remove)
    # if len(to_remove) > 0:
    #     raw.annotations.delete(to_remove)
    # # Get the events
    # events, event_id = mne.events_from_annotations(raw, event_id='auto', verbose=False)
    # shift = 0.0
    # # Epoch the data following event
    # epochs = mne.Epochs(raw, events, event_id=event_id, tmin=shift, \
    #             tmax=2.2+shift, baseline=(None, None), preload=False, verbose=False)
    # labels = epochs.events[..., -1]
    # labels -= np.min(labels)
    # data = epochs.get_data()
    # info_ep = epochs.info

    # from collections import OrderedDict
    # codes = OrderedDict()
    # for k, v in event_id.items():
    #     code = k.split('_')[0]
    #     code = code.replace('.','').replace('2','')
    #     idx = k.split('_')[1]
    #     if 'randomslowwhite' in file_name:
    #         codes[v-1] = code2array(code)
    #     else:
    #         codes[v-1] = np.array(list(map(int, code)))

    # sfreq = int(epochs.info['sfreq'])
    events = mne.find_events(raw,["stim_epoch"]) #mne.events_from_annotations(raw, event_id='auto', verbose=False)
    labels = events[..., -1]
    print(events.shape)
    n_samples_windows = int(window_size*sfreq)
    n_trial_per_class = int(len(raw)/n_class)


    n_cal = 7
    n_cal_min = 1
    n_cal_max = 10
    accuracy = []
    for n_cal in range(n_cal_min,n_cal_max):
        print("nb participant :",n_cal)
        epochs = mne.Epochs(raw,events,{"0":100,"1":101},picks=raw.ch_names[:-2],tmin=-0.01,tmax=window_size)
        X=epochs.get_data()
        print(n_class*n_cal*nb_point_per_sec)
        label = epochs.events[...,-1]-100
        size_train = int((2.2-window_size)*n_class*n_cal*nb_point_per_sec)
        print(size_train)
        X_train = X[:size_train]
        X_test = X[size_train:]
        print(X_test.shape)
        # print("test data",X_test)
        Y_train = label[:size_train]
        Y_test = label[size_train:]
        labels_test = labels_code[n_class*n_cal:]

        # window_size = 0.25
        # X_train, Y_train = to_window_old(data_train, labels_train)#, 0.25, sfreq, 60)
        # X_test, Y_test = to_window_old(data_test, labels_test)#, 0.25, sfreq, 60)

        X_std = X_train.std(axis=0)
        X_train /= X_std + 1e-8
        X_std = X_test.std(axis=0)
        X_test /= X_std + 1e-8

        # print(X_train.shape)
        # print(X_test.shape)
        # print(Y_train.shape)
        # print(len(Y_train[Y_train[:] == 1]))
        # print(Y_test.shape)
        # print(len(Y_test[Y_test[:] == 1]))
        txt = "sansBalancing"

        print("balancing the number of ones and zeros")
        txt = "avecBalancing"
        rus = RandomUnderSampler()
        counter=np.array(range(0,len(Y_train))).reshape(-1,1)
        index,_ = rus.fit_resample(counter,Y_train[:])
        X_train = np.squeeze(X_train[index,:,:], axis=1)
        Y_train = np.squeeze(Y_train[index])
        # rus = RandomUnderSampler()
        # counter=np.array(range(0,len(Y_test))).reshape(-1,1)
        # index,_ = rus.fit_resample(counter,Y_test[:])
        # X_test = np.squeeze(X_test[index,:,:], axis=1)
        # Y_test = np.squeeze(Y_test[index])


        # print(len(Y_train[Y_train[:] == 0]))
        # print(len(Y_train[Y_train[:] == 1]))
        # print(len(Y_test[Y_test[:] == 0]))
        # print(len(Y_test[Y_test[:] == 1]))

        print("Creating the different pipelines")
        start = time.time()
        clf = make_pipeline(XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="lwf"),MDM())
        time_train1 = time.time()-start
        start = time.time()
        clf2 = make_pipeline(XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="lwf"),
            TangentSpace(),
            LDA(solver="lsqr", shrinkage="auto"))
        time_train2 = time.time()-start
        start = time.time()
        clf3 = make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="lwf"),
            TangentSpace(),
            svm.SVC(C=5))
        time_train3 = time.time()-start
        start = time.time()
        clf4 = make_pipeline(Xdawn(nfilter=4, estimator="lwf"),Vectorizer(),
            LDA(solver="lsqr", shrinkage="auto"))
        time_train4 = time.time()-start


        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, shuffle=True)
        # batchsize = 128 #128 # 64 for burst
        # epochs = 45 #45 # 20 for burst

        print("fiiting the models")
        history = clf.fit(np.array(x_train), y_train)
        history2 = clf2.fit(np.array(x_train), y_train)
        history3 = clf3.fit(np.array(x_train), y_train)
        history4 = clf4.fit(np.array(x_train), y_train)
        keras.backend.clear_session()

        labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
            history.predict(X_test), dataset_moabb.codes, min_len=30, sfreq=nb_point_per_sec, consecutive=50, window_size=window_size
        )
        accuracy1 = np.round(accuracy_score(labels_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)

        labels_pred_accumul2, _, mean_long_accumul2 = make_preds_accumul_aggresive(
            history2.predict(X_test), dataset_moabb.codes, min_len=30, sfreq=nb_point_per_sec, consecutive=50, window_size=window_size
        )
        accuracy2 = np.round(accuracy_score(labels_test[labels_pred_accumul2!=-1], labels_pred_accumul2[labels_pred_accumul2!=-1]), 2)

        labels_pred_accumul3, _, mean_long_accumul3 = make_preds_accumul_aggresive(
            history3.predict(X_test), dataset_moabb.codes, min_len=30, sfreq=nb_point_per_sec, consecutive=50, window_size=window_size
        )
        accuracy3 = np.round(accuracy_score(labels_test[labels_pred_accumul3!=-1], labels_pred_accumul3[labels_pred_accumul3!=-1]), 2)

        labels_pred_accumul4, _, mean_long_accumul4 = make_preds_accumul_aggresive(
            history4.predict(X_test), dataset_moabb.codes, min_len=30, sfreq=nb_point_per_sec, consecutive=50, window_size=window_size
        )
        accuracy4 = np.round(accuracy_score(labels_test[labels_pred_accumul4!=-1], labels_pred_accumul4[labels_pred_accumul4!=-1]), 2)

        temp_accuracy = []
        line = {}
        
        print("Showing the different results")
        print("RG")
        # s11 = balanced_accuracy_score(y_train,history.predict(x_train))
        # s12 = balanced_accuracy_score(y_val,history.predict(x_val))
        # s13 = balanced_accuracy_score(Y_test,history.predict(X_test))
        # temp_accuracy.append([s11,s12,s13])
        temp_accuracy.append([0,0,accuracy1])
        # print("pred X_test :", history.predict(X_test))
        # print("pred X_train :", history.predict(X_train))

        print("RG+LDA")
        # s21 = balanced_accuracy_score(y_train,history2.predict(x_train))
        # s22 = balanced_accuracy_score(y_val,history2.predict(x_val))
        # s23 = balanced_accuracy_score(Y_test,history2.predict(X_test))
        # print("pred X_train :", history2.predict(X_train))
        # print("pred X_test :", history2.predict(X_test))
        # temp_accuracy.append([s21,s22,s23])
        temp_accuracy.append([0,0,accuracy2])

        print("RG+SVC")
        # s31 = history3.score(x_train,y_train)
        # s32 = history3.score(x_val,y_val)
        # s33 = history3.score(X_test,Y_test)
        # print("pred X_train :", history3.predict(X_train))
        # print("pred X_test :", history3.predict(X_test))
        # temp_accuracy.append([s31,s32,s33])
        temp_accuracy.append([0,0,accuracy3])

        print("Xdawn+LDA")
        # s41 = balanced_accuracy_score(y_train,history4.predict(x_train))
        # s42 = balanced_accuracy_score(y_val,history4.predict(x_val))
        # s43 = balanced_accuracy_score(Y_test,history4.predict(X_test))
        # # print("pred X_train :", history3.predict(X_train))
        # # print("pred X_test :", history3.predict(X_test))
        # temp_accuracy.append([s41,s42,s43])
        temp_accuracy.append([0,0,accuracy4])
        results_df = results_df.append({"Score":accuracy1,"time":np.mean(mean_long_accumul),"subject":id_parti,
                                        "nb_cal":n_cal,"clf":"Xdawn+RG+MDM","time_training":time_train1,
                                        "score_01":balanced_accuracy_score(Y_test,history.predict(X_test))}, ignore_index=True)
        results_df = results_df.append({"Score":accuracy2,"time":np.mean(mean_long_accumul2),"subject":id_parti,
                                        "nb_cal":n_cal,"clf":"Xdawn+RG+LDA","time_training":time_train2,
                                        "score_01":balanced_accuracy_score(Y_test,history2.predict(X_test))}, ignore_index=True)
        results_df = results_df.append({"Score":accuracy3,"time":np.mean(mean_long_accumul3),"subject":id_parti,
                                        "nb_cal":n_cal,"clf":"Xdawn+RG+SVC","time_training":time_train3,
                                        "score_01":balanced_accuracy_score(Y_test,history3.predict(X_test))}, ignore_index=True)
        results_df = results_df.append({"Score":accuracy4,"time":np.mean(mean_long_accumul4),"subject":id_parti,
                                        "nb_cal":n_cal,"clf":"Xdawn+LDA","time_training":time_train4,
                                        "score_01":balanced_accuracy_score(Y_test,history4.predict(X_test))}, ignore_index=True)

        # print(temp_accuracy)
        accuracy.append(temp_accuracy)
    accuracy = np.array(accuracy)
#     plt.figure(1)
#     # plt.plot(accuracy[:,0,0])
#     # plt.plot(accuracy[:,0,1])
#     plt.plot(range(n_cal_min,n_cal_max),accuracy[:,0,2],label=id_parti)
#     plt.legend()

#     plt.figure(2)
#     # plt.plot(accuracy[:,1,0])
#     # plt.plot(accuracy[:,1,1])
#     plt.plot(range(n_cal_min,n_cal_max),accuracy[:,1,2],label=id_parti)
#     plt.legend()

#     plt.figure(3)
#     # # plt.plot(accuracy[:,2,0])
#     # # plt.plot(accuracy[:,2,1])
#     plt.plot(range(n_cal_min,n_cal_max),accuracy[:,2,2],label=id_parti)
#     plt.legend()

#     plt.figure(4)
#     # plt.plot(accuracy[:,3,0])
#     # plt.plot(accuracy[:,3,1])
#     # print("accruacy clf 4", (accuracy[:,3,2]))
#     plt.plot(range(n_cal_min,n_cal_max),accuracy[:,3,2],label=id_parti)
#     plt.legend()    

# plt.show()

# results_df.to_csv("C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\results\\results_{0}.csv".format(txt))