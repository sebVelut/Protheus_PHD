import copy
import sys
import time
import numpy as np
import pandas as pd
import torch
import keras
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Xdawn, XdawnCovariances, Covariances
from pyriemann.classification import MDM
from pyriemann.utils.mean import mean_covariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


from SPDNet.SPD_torch.spd_net_torch import SPDNet_Module
from _utils import make_preds_accumul_aggresive
from utils import balance

from EEG2CodeKeras import basearchi
from Wavelets.Green_files.research_code.pl_utils import get_green
from Wavelets.Green_files.green.wavelet_layers import RealCovariance

from Alignments.covariance import compute_covariances
from Alignments.riemannian import compute_riemannian_alignment
from Alignments.aligner import Aligner



def get_classifier(method, window_size=None, n_channels= None):
    """
    Create the classifier depending on the method chosen

    Parameters:
    method : string, 
             Name of the classifier you want to create. It need to be in the list [CNN, SPDNet, TSLDA, TSSVC, MDM, GREEN]
    
    window_size : float, default None
                  Size in seconds of the window of the epochs. It is mandatory to create the CNN and the GREEN algorithm
    
    n_channels : int
                 Number of channels of the inputs data. It is mandatory to create the CNN and the GREEN algorithm
    """
     
    if method=="CNN":
        clf = basearchi(windows_size = window_size*500,
                        n_channel_input = n_channels)
        clf.compile(loss='binary_crossentropy',optimizer= keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
                            metrics=['accuracy'])
    elif method=="SPDNet":
        clf = make_pipeline(Covariances(estimator="lwf"),
                                        SPDNet_Module(bimap_dims=[8,4]))
    elif method=="TSLDA":
        clf = make_pipeline(Covariances(estimator="lwf"),
                            TangentSpace(), LDA(solver="lsqr", shrinkage="auto"))
    elif method=="TSSVC":
        clf = make_pipeline(Covariances(estimator="lwf"),
                            TangentSpace(), SVC())
    elif method=="MDM":
        clf = make_pipeline(Covariances(estimator="lwf"),MDM())
    elif method=="GREEN":
        clf = get_green(
                n_freqs=20,
                    kernel_width_s=window_size,
                    n_ch=n_channels,
                    sfreq=500,
                    oct_min=0,
                    oct_max=4.5,
                    orth_weights=False,
                    dropout=.7,
                    hidden_dim=[20,10],
                    logref='logeuclid',
                    pool_layer=RealCovariance(),
                    bi_out=[4],
                    dtype=torch.float32,
                    out_dim=2,
                )
    else:
        raise ValueError("choose a Classifier name that is in the list: [CNN, SPDNet, TSLDA, TSSVC, MDM, GREEN]")
     
    return clf

def train_predict(method,X_train,Y_train,X_test,Y_test,labels_codes_test,codes,subtest,prefix,n_fold=5,n_class=11,window_size=0.3,epochs=20,batchsize=64,freqwise=60):
    """
    Train with the data X_train and Y train, and predict on the X_test data. The algorithm used will be method
    The data should come from a single subject

    Parameters:
    method: str, Name of the algorithm to use to classify the 0 and 1 of the code.

    X_train: np.array, training data with shape (N_epochs, N_channels, N_samples)

    Y_train: np.array, training labels of the code (0 or 1) with shape (N_epochs,)

    X_test: np.array, testing data with shape (N_epochs, N_channels, N_samples)

    Y_test: np.array, testing labels of the code (0 or 1) with shape (N_epochs,)

    labels_codes_test: np.array, testing labels of the flickers class with shape (N_trials,)

    codes: OrderedDict(), codes of the different flickers

    subtest: int, subject whom data are coming from

    prefix: str, prefix to add to the name of the files to save

    n_fold: int, number of times to redo the train/predict process

    n_class: int, number of class of the experiment the data are coming from

    window_size: float, time in seconds of the length of the epochs

    epochs: int, number of epochs for the training

    batchsize: int, size of the batches for the training

    freqwise: int, frequence of the prediction of the bits

    return: Metrics to study the prediction
    accuracy: np.array, balanced accuracy of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,)

    recall: np.array, recall of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,)
    
    f1: np.array, f1 of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,)

    accuracy_code: np.array, balanced accuracy of the prediction on the testing data about the class of the code with shape (n_fold,)

    tps_train_code: np.array, training time of the classifier with shape (n_fold,)

    tps_test_code: np.array, prediction time of the classifier for all the testing data with shape (n_fold,)
    """
    # In
    accuracy_code = np.zeros((n_fold,1))
    tps_train_code = np.zeros((n_fold,1))
    tps_test_code = np.zeros((n_fold,1))
    accuracy = np.zeros((n_fold,1))
    recall = np.zeros((n_fold,1))
    f1 = np.zeros((n_fold,1))

    n_cal = 7
    n_channels = X_train.shape[-2]

    for k in range(n_fold):
        print("fold number ",k)
        print("Train/test of the participant : ", subtest)

        print("balancing the number of ones and zeros")
        X_train, Y_train,_= balance(X_train,Y_train,None)
        print(X_train.shape)
        print(X_test.shape)

        start = time.time()
        weight_decay = 1e-4

        # Choose the model 
        clf = get_classifier(method,window_size,n_channels)
        # Train the model
        print("Fitting")
        clf.fit(X_train,Y_train)

        print("Training finished!")
        tps_train_code[k] = time.time() - start

        # Testing
        start = time.time()
        print(X_test.shape)
        y_pred = clf.predict(X_test)
        if len(y_pred.shape)>1:
            y_pred=y_pred[:,0]

        print("getting accuracy of participant ", subtest)
        y_pred = np.array(y_pred)
        y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
        if len(Y_test.shape)>1:
            Y_test = Y_test[:,0]
        y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])


        accuracy[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)

        # get the prediction of the code's class
        labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
    y_pred_norm, codes, min_len=30, sfreq=freqwise, consecutive=50, window_size=window_size
        )
        tps_test_code[k] = time.time() - start
        accuracy_code[k] = np.round(balanced_accuracy_score(labels_codes_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
        recall[k] = recall_score(Y_test,y_pred)
        f1[k] = f1_score(Y_test,y_pred)


    # pd.DataFrame(accuracy).to_csv("./results/score/{}_score_{}_{}.csv".format(method,prefix,subtest))
    # pd.DataFrame(recall).to_csv("./results/recall/{}_recall_{}_{}.csv".format(method,prefix,subtest))
    # pd.DataFrame(f1).to_csv("./results/f1/{}_f1_{}_{}.csv".format(method,prefix,subtest))
    # pd.DataFrame(accuracy_code).to_csv("./results/score_code/{}_score_code_{}_{}.csv".format(method,prefix,subtest))
    # pd.DataFrame(tps_train_code).to_csv("./results/tps_train/{}_tps_train_{}_{}.csv".format(method,prefix,subtest))
    # pd.DataFrame(tps_test_code).to_csv("./results/tps_test/{}_tps_test_{}_{}.csv".format(method,prefix,subtest))

    return accuracy, recall, f1, accuracy_code,tps_train_code,tps_test_code

def features_preproc(X_train,Y_train,X_test,Y_test,recenter):
    """
    Preprocess the features before fitting the classifiers

    Parameters:
    X_train: np.array, training data with shape (N_epochs, N_channels, N_samples)

    Y_train: np.array, training labels of the code (0 or 1) with shape (N_epochs,)

    X_test: np.array, testing data with shape (N_epochs, N_channels, N_samples)

    Y_test: np.array, testing labels of the code (0 or 1) with shape (N_epochs,)

    recenter: boolean, Boolean to know if you recenter the data or not

    return: the preprocessed features
    X_train: np.array, preprocessed training data with shape (N_epochs, N_channels, N_samples)

    X_test: np.array, preprocessed testing data with shape (N_epochs, N_channels, N_samples)
    """
    X_preproc = np.zeros((X_train.shape[0]+X_test.shape[0],8,X_train.shape[2]))
    
    xdawn = Xdawn(nfilter=4,classes=[1],estimator='lwf')

    X_std = X_train[:X_train.shape[0]].std(axis=0)
    temp_Xtrain = X_train/(X_std + 1e-8)
    temp_Xtest = X_test/(X_std + 1e-8)
    xdawn = xdawn.fit(temp_Xtrain[:X_train.shape[0]],Y_train[:X_train.shape[0]])
    temp_Xtrain = xdawn.transform(temp_Xtrain)
    temp_Xtest = xdawn.transform(temp_Xtest)
    X_preproc[:X_train.shape[0]] = np.hstack([temp_Xtrain,np.tile(xdawn.evokeds_[None,:,:],(temp_Xtrain.shape[0],1,1))])
    X_preproc[X_train.shape[0]:] = np.hstack([temp_Xtest,np.tile(xdawn.evokeds_[None,:,:],(temp_Xtest.shape[0],1,1))])
    if recenter:
        alig = Aligner(estimator="lwf",metric="real")
        alig = alig.fit(X_preproc[:X_train.shape[0]])
        X_preproc = alig.transform(X_preproc)
    
    return X_preproc[:X_train.shape[0]],X_preproc[X_train.shape[0]:]


def train_predict_all(method,X_train,Y_train,X_test,Y_test,labels_codes_test,codes,subjects,prefix,
                      recenter=False,n_fold=5,n_class=11,window_size=0.3,epochs=20,batchsize=64,freqwise=60):
    """
    Train with the data X_train and Y train, and predict on the X_test data. The algorithm used will be method
    The data should come from a single subject

    Parameters:
    method: str, Name of the algorithm to use to classify the 0 and 1 of the code.

    X_train: np.array, training data with shape (N_epochs, N_channels, N_samples)

    Y_train: np.array, training labels of the code (0 or 1) with shape (N_epochs,)

    X_test: np.array, testing data with shape (N_epochs, N_channels, N_samples)

    Y_test: np.array, testing labels of the code (0 or 1) with shape (N_epochs,)

    labels_codes_test: np.array, testing labels of the flickers class with shape (N_trials,)

    codes: OrderedDict(), codes of the different flickers

    prefix: str, prefix to add to the name of the files to save

    n_fold: int, number of times to redo the train/predict process

    n_class: int, number of class of the experiment the data are coming from

    window_size: float, time in seconds of the length of the epochs

    epochs: int, number of epochs for the training

    batchsize: int, size of the batches for the training

    freqwise: int, frequence of the prediction of the bits

    return: Metrics to study the prediction
    accuracy: np.array, balanced accuracy of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,n_subjects)

    recall: np.array, recall of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,n_subjects)
    
    f1: np.array, f1 of the prediction on the testing data about the bits of the code (0 or 1) with shape (n_fold,n_subjects)

    accuracy_code: np.array, balanced accuracy of the prediction on the testing data about the class of the code with shape (n_fold,n_subjects)

    tps_train_code: np.array, training time of the classifier with shape (n_fold,n_subjects)

    tps_test_code: np.array, prediction time of the classifier for all the testing data with shape (n_fold,n_subjects)
    """
    n_subjects = len(subjects)
    accuracy_code = np.zeros((n_fold,n_subjects))
    tps_train_code = np.zeros((n_fold,n_subjects))
    tps_test_code = np.zeros((n_fold,n_subjects))
    accuracy = np.zeros((n_fold,n_subjects))
    recall = np.zeros((n_fold,n_subjects))
    f1 = np.zeros((n_fold,n_subjects))

    for n in range(n_subjects):
        # Feature preprocessing extraction
        X_train[n],X_test[n] = features_preproc(X_train[n],Y_train[n],X_test[n],Y_test[n],recenter)

        # Train and predict for the nth suject
        accuracy[:,n], recall[:,n], f1[:,n], accuracy_code[:,n],tps_train_code[:,n],tps_test_code[:,n] = train_predict(method,X_train[n],Y_train[n],X_test[n],Y_test[n],
                                                                                                                       labels_codes_test[n],codes,subjects[n],prefix,n_fold,
                                                                                                                       n_class,window_size,epochs,batchsize,freqwise)

    return accuracy, recall, f1, accuracy_code,tps_train_code,tps_test_code

