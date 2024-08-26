
from Alignments.riemannian import compute_riemannian_alignment
from _utils import make_preds_accumul_aggresive, make_preds_pvalue


from sklearn.model_selection import train_test_split
#from tensorflow import keras
import sys
import mne
import os
import time
import numpy as np
import pandas as pd
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn import metrics
from sklearn import svm
from scipy.stats import pearsonr
from scipy.spatial.distance import hamming 

from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

### LOADING DATA


n_cal = 7
sfreq = 500
size_win = 0.250
fps = 60
nb_consecutive_windows = 40 #number of consecutive windows with the same decision to trigger selection.  In the paper we put it to 70
min_len = 60 # context that helps for the decision process. In the paper we put it to 1
# directory = "burst100" #"mseq100"  # "set" # "burst40"   #  "mseq40"#   "dryburst100" #  # "burst40"   # "dryburst40" # "set" 
architecture ="base"# "base_patch_embbeded" #  #  "base_patch_embbeded_dilation" #  "trueeeg2code"  # "eeg2code" # " "base_noloop" #   

 
#for min_len in [40]:

    #for nb_consecutive_windows in [60]:
        
for directory in ["Dry_Ricker"]:
    path = f"C:/Users/s.velut/Documents/These/Protheus_PHD/Data/{directory}"

    nb_subject=24

    nb_class=5
    batchsize = 64 
    nb_epochs = 20

#for n_cal in range(n_cal-1, n_cal):
    table_result=np.zeros((nb_subject,5)) # this table store: subject ID, accuracy CNN training and testing, decoding time, accuracy 
    for subject in range(1,nb_subject+1):
        ###PROCESSING DATA
        file_name = f'P{subject}_dryburst100.set'
        #file_name = f'P{subject}.set'
        
        raw = mne.io.read_raw_eeglab(os.path.join(path, file_name), preload=True, verbose=False)
       
        raw = raw.resample(sfreq=500)


       
        #keep = ["EEG 001","EEG 002","EEG 003", "EEG 004", "EEG 005", "EEG 006","EEG 007","EEG 010", "EEG 008", "EEG 019"]  
        #keep = ["EEG 001","EEG 002","EEG 003", "EEG 004", "EEG 000","EEG 005"] 
        #keep = ["O1", "O2", "Oz", "P7", "P3", "P4", "P8", "Pz"] # electrodes to keep 
        #raw = raw.drop_channels([i for i in raw.ch_names if i not in keep])
        #keep = ["EEG 001","EEG 002", "EEG 003", "EEG 004","EEG 005","EEG 006"] 
        #raw = raw.drop_channels([i for i in raw.ch_names if i not in keep])
        print(raw.ch_names)

    
        

        
        raw = raw.filter(l_freq=1, h_freq=25,
                                   method="fir", verbose=True)
        # Average re-referencing
        
        #mne.set_eeg_reference(raw, 'average', copy=False, verbose=False)['EEG 005'])->good results
        #mne.set_eeg_reference(raw, ['EEG 006'])

        n_channels = len(raw.ch_names)
    
    
        


        # Strip the annotations that were script to make them easier to process
        events, event_id = mne.events_from_annotations(raw, regexp="^(?!BURST).*", event_id='auto', verbose=False)
        print(len(events))
        to_remove = []
        for idx in range(len(raw.annotations.description)):
            if (('collects' in raw.annotations.description[idx]) or
                ('iti' in raw.annotations.description[idx]) or 
                len(raw.annotations.description[idx].split('_')) != 2):
                to_remove.append(idx)
            else:
                code = raw.annotations.description[idx].split('_')[0]
                lab = raw.annotations.description[idx].split('_')[1]
                code = code.replace('\n', '')
                code = code.replace('[', '')
                code = code.replace(']', '')
                code = code.replace(' ', '')
                raw.annotations.description[idx] = code + '_' + lab

        to_remove = np.array(to_remove)
        if len(to_remove) > 0:
            raw.annotations.delete(to_remove)
        # Get the events
        events, event_id = mne.events_from_annotations(raw, regexp="^(?!BURST).*",event_id='auto', verbose=False)
        print(len(events))
        shift = 0.0
        # Epoch the data following event
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=shift, \
                    tmax=2.2+shift, baseline=(None, None), preload=False, verbose=False)
        labels = epochs.events[..., -1]
        labels -= np.min(labels)
        data = epochs.get_data()
        info_ep = epochs.info
        print("Epochs : ", len(labels))

        #####Transform Codes into np.array

        def code2array(code):
            tmp = []
            for idx, c in enumerate(code[:-2]):
                if c == '5' or c == '.':
                    continue
                elif c == '0':
                    if code[idx+2] == '5':
                        tmp.append(0.5)
                    else:
                        tmp.append(0)
                else:
                    tmp.append(1)
            if code[-1] == '.':
                if code[-2] == '0':
                    tmp.append(0)
                else:
                    tmp.append(1)
            return np.array(tmp)


        #BUILL DICTIONNARY OF CODES
        from collections import OrderedDict
        codes = OrderedDict()
        for k, v in event_id.items():
            code = k.split('_')[0]
            code = code.replace('.','').replace('2','')
            idx = k.split('_')[1]
            if 'randomslowwhite' in file_name:
                codes[v-1] = code2array(code) 
            else:
                codes[v-1] = np.array(list(map(int, code)))

        #####DEFINE TRAIN/TEST SPLIT AND WINDOW SIZE

        sfreq = int(epochs.info['sfreq'])
        
        n_samples_windows = int(size_win*sfreq)

        n_trial_per_class = int(len(data) / nb_class)

        data = compute_riemannian_alignment(data, mean=None, dtype='real')


        data_train = data[:nb_class * n_cal]
        labels_train = labels[:nb_class* n_cal]
        data_test = data[nb_class * n_cal:]
        labels_test = labels[nb_class * n_cal:]


        ### Slice the epoch in windows
        #The network is not processing full epochs but windows of 250ms. 
        #So each epoch is cut into window and the following code (`0` or `1`) is associated as label.


        def to_window(data, labels):
            length = int((2.2-size_win)*sfreq)
            X = np.empty(shape=((length)*data.shape[0], n_channels, 
        n_samples_windows))
            y = np.empty(shape=((length)*data.shape[0]), dtype=int)
            print(length)
            print(n_samples_windows)
            count = 0
            for trial_nb, trial in enumerate(data):
                lab = labels[trial_nb]
                c = codes[lab]
                code_pos = 0
                for idx in range(length):
                    X[count] = trial[:, idx:idx+n_samples_windows]
                    if idx/sfreq >= (code_pos+1)/fps:
                        code_pos += 1
                    y[count] = int(c[code_pos])
                    count += 1

            # X = np.expand_dims(X, 1)
            X = X.astype(np.float32)
            y_pred = np.vstack((y,np.abs(1-y))).T
            y = np.array([1 if (y >= 0.5) else 0 for y in y_pred[:,0]])
            return X, y

        X_train, y_train = to_window(data_train, labels_train)
        X_test, y_test = to_window(data_test, labels_test)

        print(data_train.shape)

        #NORMALIZATION USING DATA FROM TRAIN
        X_std = X_train.std(axis=0)
        X_train /= X_std + 1e-8

        X_test /= X_std + 1e-8

       



        ### Balance classes
        #Our classes are unbalanced, there are more `1` than `0` in the train set
        #  (the stimulation is more often ON than OFF).  
        #We will use a random under sampler to make it balance.


       

        rus = RandomUnderSampler()
        counter=np.array(range(0,len(y_train))).reshape(-1,1)
        index,_ = rus.fit_resample(counter,y_train[:])
        X_train = np.squeeze(X_train[index,:,:], axis=1)
        y_train = np.squeeze(y_train[index])


        print("XTrain shape:", X_train.shape)
        print("yTrain shape:", y_train.shape)
        print("XTest shape :", X_test.shape)
        print("yTest shape :", y_test.shape)

      

    #PICK AN ARCHITECTURE

        #clf = make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="scm"),MDM())
        
        #parameters = {'nfilter': [1],'estimator':['lwf'],'xdawn_estimator':['scm'],scoring"accuracy"}
        #clf = make_pipeline(GridSearchCV(XdawnCovariances(),parameters), TangentSpace(), LDA(solver="lsqr", shrinkage="auto"))
        clf = make_pipeline(XdawnCovariances(nfilter=8, estimator="lwf", xdawn_estimator="lwf",classes=[1]), TangentSpace(), LDA(solver="lsqr", shrinkage="auto"))
        
        #clf = make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="scm"), TangentSpace(), svm.SVC())

        #clf = make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="scm"), TangentSpace(), RandomForestClassifier(max_depth=4, random_state=0))

        #parameters = {'solver': ['lbfgs'], 'max_iter': [2000], 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3]}
        #clf = make_pipeline(XdawnCovariances(nfilter=4, estimator="lwf", xdawn_estimator="scm"), TangentSpace(), GridSearchCV(MLPClassifier(), parameters, n_jobs=-1))

        #clf.summary()


        #CUT THE TRAIN IN TRAIN AND VALID
        X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42, shuffle=True)


        X_train.shape


        ### Attach an optimizer and train the network

    

        #lr = 1e-3
        #weight_decay = 1e-4
        #optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)


        #fitting_time = time.time() #initiate the clock to compute CNN training time
        #clf.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        #history = clf.fit(X_train, y_train, batch_size=batchsize, epochs=nb_epochs, validation_data=(x_val, y_val), shuffle=True)
        history = clf.fit(np.array(X_train), y_train)
        #keras.backend.clear_session()

        #fitting_time = time.time()-fitting_time



        #PREDICTION ON THE TEST SET

        # keras.backend.clear_session()
        #y_pred = clf.predict(X_test)[:,0]
        y_pred = clf.predict(X_test)
        y_pred = np.array(y_pred)

        y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in y_pred])
        y_test_norm = np.array([0 if y == 0 else 1 for y in y_test])




        ### Compute accuracy score and accuracy score when a prediction is made (discard not classified trials)
        #print(len(labels_pred[labels_pred!=-1])/len(labels_pred))
        #print(accuracy_score(labels_test[labels_pred!=-1], labels_pred[labels_pred!=-1]))


        ###### Mean Classification time


        ### Other classification method
        #Same as before but the classification method is different. Instead of thresholds to reach,
        #if when increasing trial lengt a code correllated the most "nb_consecutive_windows" times in a row then the trial is labeled.
        #min len = 50 , consecutive windows =60 BURST


        labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                        y_pred_norm, codes, min_len=min_len, sfreq=sfreq,consecutive=nb_consecutive_windows
                    )


        print(len(labels_pred_accumul[labels_pred_accumul!=-1])/len(labels_pred_accumul))
        print(f"Prediction score is: {balanced_accuracy_score(labels_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1])}")

        classif_time=np.mean( mean_long_accumul)
        nb_of_prediction=len(labels_pred_accumul[labels_pred_accumul!=-1])/len(labels_pred_accumul)
        accuracy=accuracy_score(labels_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1])

        table_result[subject-1,0]=history.score(X_train,y_train)
        table_result[subject-1,1]=history.score(x_val,y_val)
        table_result[subject-1,2]=history.score(X_test,y_test)
        table_result[subject-1,3]=classif_time
        table_result[subject-1,4]=accuracy
        #np.savetxt(f"./results_NER2/{directory}_nb_calibration_{n_cal}_per_subject_{architecture}.csv", table_result, delimiter=",")
        np.savetxt(f"./Results/{directory}_nb_calibration_{n_cal}_per_subject_{architecture}_{min_len}_{nb_consecutive_windows}.csv", table_result, delimiter=",")


        table_stat = np.array([np.mean(table_result[:,0]),np.std(table_result[:,0]),
                            np.mean(table_result[:,1]),np.std(table_result[:,1]),
                            np.mean(table_result[:,2]),np.std(table_result[:,2]),
                            np.mean(table_result[:,3]),np.std(table_result[:,3]),
                            np.mean(table_result[:,4]),np.std(table_result[:,4])])
        #np.savetxt(f"./results/{directory}_nb_calibration_{n_cal}_stat_{architecture}.csv", table_stat, delimiter=",")
        np.savetxt(f"./Results/{directory}_nb_calibration_{n_cal}_stat_{architecture}_{min_len}_{nb_consecutive_windows}.csv", table_stat, delimiter=",")


        print(f'Cnn train acc: {np.mean(table_result[:,0])} Cnn val acc: {np.mean(table_result[:,1])} Mean CNN time: {np.mean(table_result[:,2])} Mean decoding time: {np.mean(table_result[:,3])}  Mean accuracy:  {np.mean(table_result[:,4])}')
        print(f'Cnn train acc: {np.std(table_result[:,0])}  Std Cnn val acc: {np.std(table_result[:,1])} Std CNN time: {np.std(table_result[:,2])} Std decoding time: {np.std(table_result[:,3])}  Std accuracy:  {np.std(table_result[:,4])}')
        
        
        labels_pred_accumul

        labels_test
