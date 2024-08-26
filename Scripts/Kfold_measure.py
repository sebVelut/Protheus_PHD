


import copy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from _utils import make_preds_accumul_aggresive

from tensorflow import keras
from EEG2CodeKeras import basearchi
from SPDNet.tensorflow.spd_net_tensorflow import SPDNet_Tensorflow
from SPDNet.torch.spd_net_bn_torch import SPDNetBN_Module

from utils import balance


class Kfolder():
    def __init__(self,clf_name,n_fold,epochs=20,batch_size=64,method=["SS","DG","DA"]):
        self.epochs = epochs
        self.batch_size = batch_size
        self.method = method
        self.clf_name = clf_name
        self.n_fold = n_fold

    def DG_Kfold(self,X_parent,Y_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        
        accuracy_code_DG = np.zeros((self.n_fold,1))
        tps_train_code_DG = np.zeros((self.n_fold,1))
        tps_test_code_DG = np.zeros((self.n_fold,1))
        accuracy_DG = np.zeros((self.n_fold,1))

        for k in range(self.n_fold):
            print("Fold number ",k)
            print("TL to the participant : ", subtest)
            ind2take = [j for j in range(len(subjects)) if j!=subtest]
            X = X_parent.copy()
            Y = Y_parent.copy()

            X_train = np.concatenate(X[ind2take])
            Y_train = np.concatenate(Y[ind2take])

            X_test = X[subtest]
            Y_test = Y[subtest]
            labels_code_test = labels_codes[subtest]

            print("balancing the number of ones and zeros")
            X_train, Y_train,_= balance(X_train,Y_train,None)
            print(X_train.shape)
            print(X_test.shape)

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4

            # Choose the model 
            classifiers = {
                        "CNN": basearchi(windows_size = X_parent[0].shape[-1],
                                         n_channel_input = X_parent[0].shape[-2]),
                        "SPD": SPDNet_Tensorflow(bimap_dims=[28,14,7]),
                        "SPDBN": SPDNetBN_Module(bimap_dims=[32,28,14,7]),
                        }
            clf = classifiers.get(self.clf_name)
            if self.clf_name == "CNN":
                clf.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
                            metrics=['accuracy'])
            # Train the model
            clf.fit(X_train,Y_train,epochs=self.epochs)

            print("Training finished!")
            tps_train_code_DG[k] = time.time() - start

            # Testing
            start = time.time()
            y_pred = clf.predict(X_test)
            if len(y_pred.shape)>1:
                y_pred=y_pred[:,0]
            print("getting accuracy of participant ", subtest)
            y_pred = np.array(y_pred)
            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
            if len(Y_test.shape)>1:
                Y_test = Y_test[:,0]
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

            accuracy_DG[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_DG[k] = time.time() - start
            accuracy_code_DG[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)


        pd.DataFrame(accuracy_DG).to_csv("./results/score/{}_DG_score_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(accuracy_code_DG).to_csv("./results/score_code/{}_DG_score_code_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_train_code_DG).to_csv("./results/tps_train/{}_DG_tps_train_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_test_code_DG).to_csv("./results/tps_test/{}_DG_ps_test_{}_{}.csv".format(self.clf_name,prefix,subtest))
        return 0

    def Normal_Kfold(self,X_parent,Y_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        accuracy_code_SS = np.zeros((self.n_fold,1))
        tps_train_code_SS = np.zeros((self.n_fold,1))
        tps_test_code_SS = np.zeros((self.n_fold,1))
        accuracy_SS = np.zeros((self.n_fold,1))

        n_cal = 7
        nb_samples_windows = int((2.2-window_size)*n_class*n_cal*60)

        for k in range(self.n_fold):
            print("fold number ",k)
            print("TL to the participant : ", subtest)
            X = X_parent.copy()
            Y = Y_parent.copy()

            X_train = X[subtest][:nb_samples_windows]
            Y_train = Y[subtest][:nb_samples_windows]

            X_test = X[subtest][nb_samples_windows:]
            Y_test = Y[subtest][nb_samples_windows:]
            labels_code_test = labels_codes[subtest][n_cal*n_class:]

            print("balancing the number of ones and zeros")
            X_train, Y_train,_= balance(X_train,Y_train,None)
            print(X_train.shape)
            print(X_test.shape)

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4

            # Choose the model 
            classifiers = {
                        "CNN": basearchi(windows_size = X_train.shape[-1],
                                         n_channel_input = X_train.shape[-2]),
                        "SPD": SPDNet_Tensorflow(bimap_dims=[28,14,7]),
                        "SPDBN": SPDNetBN_Module(bimap_dims=[32,28,14,7]),
                        }
            clf = classifiers.get(self.clf_name)
            if self.clf_name == "CNN":
                clf.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
                            metrics=['accuracy'])
            # Train the model
            clf.fit(X_train,Y_train,epochs=self.epochs,batch_size=self.batch_size,shuffle=True)

            print("Training finished!")
            tps_train_code_SS[k] = time.time() - start

            # Testing
            start = time.time()
            y_pred = clf.predict(X_test)
            if len(y_pred.shape)>1:
                y_pred=y_pred[:,0]

            print("getting accuracy of participant ", subtest)
            y_pred = np.array(y_pred)
            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
            if len(Y_test.shape)>1:
                Y_test = Y_test[:,0]
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])
            

            accuracy_SS[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_SS[k] = time.time() - start
            accuracy_code_SS[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)


        pd.DataFrame(accuracy_SS).to_csv("./results/score/{}_SS_score_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(accuracy_code_SS).to_csv("./results/score_code/{}_SS_score_code_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_train_code_SS).to_csv("./results/tps_train/{}_SS_tps_train_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_test_code_SS).to_csv("./results/tps_test/{}_SS_tps_test_{}_{}.csv".format(self.clf_name,prefix,subtest))
        return 0

    def DA_Kfold(self,X_parent,Y_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        accuracy_code_DA = np.zeros((self.n_fold,1))
        tps_train_code_DA = np.zeros((self.n_fold,1))
        tps_test_code_DA = np.zeros((self.n_fold,1))
        accuracy_DA = np.zeros((self.n_fold,1))

        n_cal = 4
        nb_samples_windows = int((2.2-window_size)*n_class*n_cal*60)

        for k in range(self.n_fold):
            print("fold number ",k)
            print("TL to the participant : ", subtest)
            ind2take = [j for j in range(len(subjects)) if j!=subtest]
            X = X_parent.copy()
            Y = Y_parent.copy()

            X_train = np.concatenate([np.concatenate(X[ind2take]),X[subtest][:nb_samples_windows]])
            Y_train = np.concatenate([np.concatenate(Y[ind2take]),Y[subtest][:nb_samples_windows]])

            X_test = X[subtest][nb_samples_windows:]
            Y_test = Y[subtest][nb_samples_windows:]
            labels_code_test = labels_codes[subtest][n_cal*n_class:]

            print("balancing the number of ones and zeros")
            X_train, Y_train,_= balance(X_train,Y_train,None)
            print(X_train.shape)
            print(Y_train.shape)
            print(X_test.shape)

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4

            # Choose the model 
            classifiers = {
                        "CNN": basearchi(windows_size = X_parent[0].shape[-1],
                                         n_channel_input = X_parent[0].shape[-2]),
                        "SPD": SPDNet_Tensorflow(bimap_dims=[28,14,7]),
                        "SPDBN": SPDNetBN_Module(bimap_dims=[32,28,14,7]),
                        }
            clf = classifiers.get(self.clf_name)
            if self.clf_name == "CNN":
                clf.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
                            metrics=['accuracy'])
            # Train the model
            clf.fit(X_train,Y_train,epochs=self.epochs)

            print("Training finished!")
            tps_train_code_DA[k] = time.time() - start

            # Testing
            start = time.time()
            y_pred = clf.predict(X_test)
            if len(y_pred.shape)>1:
                y_pred=y_pred[:,0]
            print("getting accuracy of participant ", subtest)
            y_pred = np.array(y_pred)
            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
            if len(Y_test.shape)>1:
                Y_test = Y_test[:,0]
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

            accuracy_DA[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_DA[k] = time.time() - start
            accuracy_code_DA[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)


        pd.DataFrame(accuracy_DA).to_csv("./results/score/{}_DA_score_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(accuracy_code_DA).to_csv("./results/score_code/{}_DA_score_code_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_train_code_DA).to_csv("./results/tps_train/{}_DA_tps_train_{}_{}.csv".format(self.clf_name,prefix,subtest))
        pd.DataFrame(tps_test_code_DA).to_csv("./results/tps_test/{}_DA_tps_test_{}_{}.csv".format(self.clf_name,prefix,subtest))
        return 0
    
    def perform_Kfold(self,X,Y,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        if "SS" in self.method:
            print("Get score without transfer learning \n\n")
            self.Normal_Kfold(X,Y,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)
        if "DG" in self.method:
            print("Get score with DG transfer learning \n\n")
            self.DG_Kfold(X,Y,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)
        if "DA" in self.method:
            print("Get score with DA transfer learning \n\n")
            self.DA_Kfold(X,Y,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)

        return 0

