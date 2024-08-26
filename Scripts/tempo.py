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
from utils import prepare_data,get_BVEP_data,balance,get_y_pred


from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from sklearn.cross_decomposition import CCA
import keras

from pyriemann.estimation import XdawnCovariances, Xdawn
from sklearn.manifold import TSNE
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann, distance
from pyriemann.utils.utils import check_weights
from pyriemann.utils.base import powm
from pyriemann.tangentspace import TangentSpace
from mne.decoding import Vectorizer
from pyriemann.transfer import (
    decode_domains,
    encode_domains,
    TLCenter,
    TLStretch,
    TLRotate,
)

from tensorflow import keras
from collections import OrderedDict
import tensorflow as tf
import mne
import time
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import CVEP, MotorImagery, LeftRightImagery
from SPDNet.tensorflow.spd_net_2_tensorflow import SPDNet_AJD
from SPDNet.tensorflow.spd_net_tensorflow import SPDNet_Tensorflow
# from SPDNet.tensorflow.optimizer import riemannian_adam
from SPDNet.torch.optimizers import riemannian_adam as torch_riemannian_adam
from SPDNet.torch.spd_net_bn_torch import SPDNetBN_Torch, SPDNetBN_Module, CNNSPDNetBN_Module
from DNorm_CLF.DNorm_SPD.DNorm_SPD import BNSPD_Net
from sklearn.model_selection import GridSearchCV
import moabb
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts\\SPDNet")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\riemannian_tSNE")
from R_TSNE import R_TSNE
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40
from liu2024 import Liu2024
# get the functions from RPA package
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\RPA")
import rpa.transfer_learning as TL
import rpa.diffusion_map as DM
import rpa.get_dataset as GD

# fps = 60
# sfreq = 500
# window_size=0.25
# n_class=4
# n_samples_windows = int(window_size*sfreq)

# subjects = [1,2,3]
# # subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
# n_channels = 32
# on_frame = True
# tospd = False
# recenter = True
# if on_frame:
#     freq = fps
# else:
#     freq = sfreq

# raw_data,labels,codes,labels_codes = get_BVEP_data(subjects,on_frame)
# X_parent,Y_parent,domains_parent = prepare_data(subjects,raw_data,labels,on_frame,tospd,recenter,codes=codes)

# n_cal = 7
# n_class = 4
# nb_fold = 1
# spd_accuracy_code_perso = np.zeros((nb_fold,12))
# spd_tps_train_code_perso = np.zeros((nb_fold,12))
# spd_tps_test_code_perso = np.zeros((nb_fold,12))
# spd_accuracy_perso = np.zeros((nb_fold,12))

# for k in range(nb_fold):
#     for i in range(len(subjects)):
#         print("TL to the participant : ", i)
#         ind2take = [j for j in range(len(subjects)) if j!=i]

#         X = X_parent.copy()
#         Y = Y_parent.copy()
#         domains = domains_parent.copy()
#         nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*fps)

#         ## For SS
#         # X_train = X[i][:nb_sample_cal]
#         # Y_train = Y[i][:nb_sample_cal]
#         # X_test = X[i][nb_sample_cal:]
#         # Y_test = Y[i][nb_sample_cal:]
#         # labels_code_test = labels_codes[i][(n_class*n_cal):]

#         ## For DA
#         X_train = np.concatenate([np.concatenate(X[ind2take]).reshape(-1,X.shape[-2],X.shape[-1]),X[i][:nb_sample_cal]]).reshape(-1,X.shape[-2],X.shape[-1])
#         Y_train = np.concatenate([np.concatenate(Y[ind2take]).reshape(-1),Y[i][:nb_sample_cal]]).reshape(-1)
#         X_test = X[i][nb_sample_cal:]
#         Y_test = Y[i][nb_sample_cal:]
#         labels_code_test = labels_codes[i][n_cal*n_class:]

#         print(X_train.shape)
#         print(X_test.shape)
#         # X_std = X_train.std(axis=0)
#         # X_train /= X_std + 1e-8
#         # X_std = X_test.std(axis=0)
#         # X_test /= X_std + 1e-8

#         print("balancing the number of ones and zeros")
#         rus = RandomUnderSampler()
#         counter=np.array(range(0,len(Y_train))).reshape(-1,1)
#         index,_ = rus.fit_resample(counter,Y_train[:])
#         X_train = np.squeeze(X_train[index,:,:], axis=1)
#         Y_train = np.squeeze(Y_train[index])

#         print("Creating the different pipelines")
#         lr = 1e-3
#         # optimizer = riemannian_adam.RiemannianAdam(learning_rate=lr)
#         batchsize = 64 #128 # 64 for burst
#         epoch = 20 #45 # 20 for burst
#         # clf = SPDNet_AJD(n_epochs=epoch,batch_size=batchsize,valid_split=0.1)
#         clf = CNNSPDNetBN_Module(32,bimap_dims=[17,16,8,4])
#         # clf = SPDNetBN_Module(32,0.25,bimap_dims=[28,14,7])

#         print("Fitting")
#         start = time.time()
#         weight_decay = 1e-4
        
#         x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42, shuffle=True)
#         history = clf.fit(np.array(x_train), y_train,
#                         batch_size=batchsize, epochs=epoch)
#         spd_tps_train_code_perso[k][i] = time.time() - start
        
#         print("getting accuracy of participant ", i)
#         start = time.time()
#         y_pred = clf.predict(X_test)
#         y_pred = np.array(y_pred)
#         y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
#         y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

#         tn, fp, fn, tp = confusion_matrix(y_test_norm, y_pred_norm).ravel()
#         spd_accuracy_perso[k][i] = balanced_accuracy_score(y_test_norm,y_pred_norm)

#         labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
#             y_pred_norm, codes, min_len=30, sfreq=fps, consecutive=50, window_size=window_size
#         )
#         print("meaaaaaaaaaaaaaan of long",np.mean(mean_long_accumul))
#         spd_tps_test_code_perso[k][i] = time.time() - start
#         spd_accuracy_code_perso[k][i] = np.round(accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
#         keras.backend.clear_session()

# spd_accuracy_perso = np.mean(spd_accuracy_perso,axis=0)
# spd_tps_train_code_perso = np.mean(spd_tps_train_code_perso,axis=0)
# spd_tps_test_code_perso = np.mean(spd_tps_test_code_perso,axis=0)
# spd_accuracy_code_perso = np.mean(spd_accuracy_code_perso,axis=0)

# print(spd_accuracy_perso)
# print(spd_tps_train_code_perso)
# print(spd_tps_test_code_perso)
# print(spd_accuracy_code_perso)
# np.save("C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/Score_TF/SPDNet/WO_score",spd_accuracy_perso)
# np.save("C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/Score_TF/SPDNet/WO_score_code",spd_accuracy_code_perso)
# np.save("C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/Score_TF/SPDNet/WO_tps_train_code",spd_tps_train_code_perso)
# np.save("C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/Score_TF/SPDNet/WO_tps_test_code",spd_tps_test_code_perso)


#################### from csv per participant to global

# prefix_file = ['CNN_DA_score_code_recentered_','CNN_DG_score_code_recentered_','CNN_SS_score_code_recentered_']
# path_file = 'C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/Score_TF/score_code/'

# nb_subject= 12

# df_results = {}

# for pf in prefix_file:
#     for i in range(nb_subject):
#         df_sub = pd.read_csv(path_file+pf+str(i)+".csv")
#         df_results[str(i)] = df_sub.values[:,1]

#     pd.DataFrame(df_results).to_csv(path_file+pf[:-1]+".csv")

#################### test liu dataset of Taha

ds = Liu2024()
paradigm = LeftRightImagery()


print("\ntest\n")

raw,lab,met = paradigm.get_data(ds,[1])
# print(raw[0])
# events = mne.events_from_annotations(raw[0])
# print(events)
# print(raw[0].get_data().shape)

print(lab)
Elab = lab
# Y = Elab[Elab>=3]
# X = raw[0].get_data()[Elab>=3]
# print(X.shape)
# print(Y.shape)

# xtrain,y_train,x_test,y_test = train_test_split(X,Y,test_size=0.2)

# clf  = LDA()
clf = make_pipeline(Xdawn(nfilter=4, estimator="lwf"),Vectorizer(),
            LDA(solver="lsqr", shrinkage="auto"))

pipes = {}
pipes["LDA"] = clf
# pipes["LDA"] = make_pipeline(clf)

ds.subject_list = ds.subject_list[:2]

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=ds,
    suffix="braindecode_example",
    overwrite=True,
    return_epochs=False,
    n_jobs=1,
)

results = evaluation.process(pipes)

print(results.head())