from datetime import datetime
import pickle
from matplotlib import pyplot as plt
import mne
import argparse
import sys
sys.path.append('D:/s.velut/Documents/Thèse/Protheus_PHD/Scripts')
from EEG2CodeKeras import EEG2Code
from Wavelets.Green_files.green.wavelet_layers import RealCovariance
import torch

from Wavelets.Green_files.research_code.pl_utils import get_green
mne.set_log_level('ERROR')
from utils import balance
from _utils import make_preds_accumul_aggresive
sys.path.insert(0,"D:/s.velut/Documents/Thèse/Protheus_PHD/Scripts/SPDNet/")
import numpy as np
import time
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from SPDNet.SPD_torch.optimizers import riemannian_adam as torch_riemannian_adam
from Alignments.covariance import compute_covariances
from Alignments.riemannian import compute_riemannian_alignment,compute_ref_riemann
from Alignments.aligner import Aligner
from torch.utils.data import DataLoader, TensorDataset
import Similarity_measure as simim
from STLDataLoader import STLDataLoader
from SPDNet.SPD_torch.spd_net_torch import SPDNet_Module
from xdawn_supertrial import XdawnST
import keras

torch.set_default_device('cpu')


def train(clf,X_train, Y_train, test_size=0.2, batchsize=64, lr=1e-3, num_epochs=20, device=torch.device('cpu')):
    #fit the classifier
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=test_size, random_state=42, shuffle=True)

    # Convert data into PyTorch tensors
    X_train_tensor = torch.tensor(x_train, dtype=torch.float64,device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long,device=device)
    X_val_tensor = torch.tensor(x_val, dtype=torch.float64,device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long,device=device)

    # Create DataLoader for train, validation, and test sets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True,generator=torch.Generator(device='cpu'))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False,generator=torch.Generator(device='cpu'))

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch_riemannian_adam.RiemannianAdam(clf.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_y_pred= []
        y_train = []
        clf.train()
        for inputs, labels in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = clf(inputs)
            labels = labels.to('cpu')
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            train_y_pred.append(predicted.to('cpu'))
            y_train.append(labels.to('cpu'))

            running_loss += loss.item()
        train_accuracy = balanced_accuracy_score(np.concatenate(y_train),np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(train_y_pred)]))

        # Validation
        clf.eval()
        val_correct = 0
        val_y_pred = []
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = clf(inputs)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to('cpu')
                val_correct += (predicted == labels).sum().item()
                val_y_pred.append(predicted)


            val_accuracy = balanced_accuracy_score(y_val,np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(val_y_pred)]))
            print(f"Epoch {epoch+1} train Accuracy: {train_accuracy} ||  Validation Accuracy: {val_accuracy}")

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

    #print("Training finished!")
    #print("Training accuracy :", train_accuracy)

    return clf

def predict(clf, X_test, batchsize=64,device=torch.device('cpu')):
    #get the different results
    # Convert data into PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64,device=device)

    # Create DataLoader for train, validation, and test sets
    test_dataset = TensorDataset(X_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False,generator=torch.Generator(device='cpu'))
    
    # Testing
    y_pred= []
    with torch.no_grad():
        for inputs in test_dataloader:
            outputs = clf(inputs[0])
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.to('cpu')
            y_pred.append(predicted)

    test_y_pred = np.concatenate(y_pred)
    y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in test_y_pred])

    return y_pred_norm

def get_all_metrics(Y_test, Y_pred, code_test, code_pred):

    score = balanced_accuracy_score(Y_test,Y_pred)
    recall = recall_score(Y_test,Y_pred)
    f1 = f1_score(Y_test,Y_pred)

    score_code = balanced_accuracy_score(code_test,code_pred)

    return score, recall, f1, score_code

def full_preprocessed_data(path,participant):
    return np.load(path+"full_preprocess_data_"+participant+".npy")


def solo_preprocessed_data(path,participant):
    return np.load(path+"cal_2/full_solo_preprocess_data_"+participant+".npy")


def similarity_score(Xt, Yt, Xs, Ys):
    sim = {}

    print("Calcul of spearman correlation")
    sim["Spearman Correlation"] = simim.spearman_correlation_similarity(Xt, Yt, Xs, Ys)
    print("Calcul of pearson correlation")
    sim["Pearson Correlation"] = simim.pearson_correlation_similarity(Xt, Yt, Xs, Ys)
    print("Calcul of cosine similarity")
    sim["Cosine Similarity"] = simim.cosine_similarity(Xt,Yt,Xs,Ys)
    print("Calcul of riemannian similarity")
    sim["Riemannian Similarity"] = simim.riemannian_distance(Xt,Yt,Xs,Ys)
    print("Calcul of euclidian similarity")
    sim["Euclidian Similarity"] = simim.euclidian_distance(Xt,Yt,Xs,Ys)

    return sim

def perform_measure(clf, X_train, Y_train, X_test, Y_test, codes, n_class=5, n_cal=4, window_size=0.35, freqwise=500,
                    test_size=0.2, batchsize= 64, lr=1e-3, num_epochs=20, device=torch.device("cpu")):
    # Initialisation
    nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

    # Get the training data 
    temp_start = time.time()

    clf = train(clf, X_train, Y_train, test_size, batchsize, lr, num_epochs, device)
    tps_train = time.time() - temp_start

    temp_start = time.time()
    Y_pred = predict(clf, X_test, batchsize, device)
    tps_pred = time.time() - temp_start

    labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
            Y_pred, codes, min_len=30, sfreq=freqwise, consecutive=50, window_size=window_size
        )
    
    tps_acc = np.mean(mean_long_accumul)

    return Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc

def perform_measure_TF(clf, X_train, Y_train, X_test, Y_test, codes, n_class=5, n_cal=4, window_size=0.35, freqwise=500,
                    test_size=0.2, batchsize= 64, lr=1e-3, num_epochs=20, device=torch.device("cpu")):
    # Initialisation
    nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

    # Get the training data 

    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

    print("Fitting")
    start = time.time()
    clf = clf.fit(np.array(x_train), y_train)

    tps_train = time.time() - start

    temp_start = time.time()
    y_pred = clf.predict(X_test)
    y_pred = np.array(y_pred)
    Y_pred = np.array([1 if (y >= 0.5) else 0 for y in y_pred])

    tps_pred = time.time() - temp_start

    labels_pred_accumul, code_buffer, mean_long_accumul = make_preds_accumul_aggresive(
            Y_pred, codes, min_len=30, sfreq=freqwise, consecutive=50, window_size=window_size
        )
    
    # to_save = {}
    # to_save["y_pred"] = Y_pred
    # to_save["codes"] = codes
    # to_save["min_len"] = 30
    # to_save["sfreq"] = freqwise
    # to_save["cons"] = 50
    # to_save["window_size"] = window_size
    # to_save["labels_pred"] = labels_pred_accumul
    # to_save["code_buffer"] = code_buffer
    # to_save["mean_long"] = mean_long_accumul

    # print(to_save)

    # now = datetime.now().strftime("%Y%m%d-%H%M%S")
    # with open(f"D:/s.velut/Documents/Thèse/Juan Help/burst-offline-benchmark/results/decoding_accumulation_{now}_{0}.json", 'wb') as f:
    #     pickle.dump(to_save, f)
    
    tps_acc = np.mean(mean_long_accumul)

    return Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc

def get_train_test_data(Xt, Yt, Xs, Ys, domainst, domainss, codes, labels_code, method, clf_name, n_class=5, n_cal=4, window_size=0.35, freqwise=500,
                    test_size=0.2):
    if method=="SiSu":
        # Initialisation
        nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

        # Get the training data 
        X_train = Xt[:nb_sample_cal]
        Y_train = Yt[:nb_sample_cal]
        domains_train = domainst[:nb_sample_cal]
        X_test = Xt[nb_sample_cal:]
        Y_test = Yt[nb_sample_cal:]
        labels_code_test = labels_code[(n_class*n_cal):]
        if clf_name in ["TS_LDA","TS_SVM","MDM","SPD","CNN","GREEN","DACNN"]:
            X_std = Xt[:nb_sample_cal].std(axis=0)
            X_train = X_train/(X_std + 1e-8)
            X_test = X_test/(X_std + 1e-8)

        X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)
    elif method=="DA":
        # Initialisation
        nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)

        # Get the training data 
        X_train = np.concatenate([Xs,Xt[:nb_sample_cal]]).reshape(-1,Xs.shape[-2],Xs.shape[-1])
        Y_train = np.concatenate([Ys,Yt[:nb_sample_cal]]).reshape(-1)
        domains_train = np.concatenate([domainss,domainst[:nb_sample_cal]]).reshape(-1)
        X_test = Xt[nb_sample_cal:]
        Y_test = Yt[nb_sample_cal:]
        labels_code_test = labels_code[(n_class*n_cal):]
        if clf_name in ["TS_LDA","TS_SVM","MDM","SPD","CNN","GREEN","DACNN"]:
            X_std = Xt[:nb_sample_cal].std(axis=0)
            X_train = X_train/(X_std + 1e-8)
            X_test = X_test/(X_std + 1e-8)

        X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)

    return X_train, Y_train, X_test, Y_test, labels_code_test


def main(path, file_path, fmin, fmax, sample_freq, fps, timewise, participants, clf_name,
         method="DA", window_size=0.35, test_size=0.2, batchsize=64, lr=1e-3, num_epochs=20, prefix=''):
    print("device cpu ?",torch.cpu.device_count())
    ##### Here is to centralised main steps
    participants = eval(participants)

    data_path = '/'.join([file_path,"ws"+str(window_size),timewise,"data",''])

    # get the data
    dl = STLDataLoader(path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, 5)
    raw_data = dl.load_data()
    X, Y, domains, codes, labels_code = dl.get_epochs(raw_data)

    # Initialisation
    similarity = {}
    all_metrics = np.zeros((8,dl.nb_subjects, dl.nb_subjects))
    # Start for loop on each participant (or just on a few to go faster)
    for i in range(dl.nb_subjects):
        print("Check participant ",i)
        similarity[participants[i]] = {}
        
        # Preprocess the data for data i with no data leak(index of the loop). HERE IF DOMAIN ADAPTATION
        if method in ["DA","SiSu"]:
            print("Preprocess the data of participant ",i)
            temp_start = time.time()
            Xt_preproc = solo_preprocessed_data(data_path,participants[i])
            tps_preproc = time.time() - temp_start
        else:
            Xt_preproc=None
        
        # Start another for loop to perform 2by2 measure (similarity, Domain Adaptation/Generalisation)
        for j in range(dl.nb_subjects):
            print("With participant", j)
            if (method!="SiSu" and j!=i) or (method=="SiSu" and j==i):
                # Preprocess the data for data i with no data leak(index of the loop) HERE IF DOMAIN GENERALISATION
                # if method=="DG":
                #     print("Preprocess the data of participant ",i)
                #     temp_start = time.time()
                #     Xt_preproc = soloPreprocess(X[i], Y[i],X[j], Y[j], method="DG", n_class=dl.n_class, n_cal=4, window_size=dl.window_size,freqwise=dl.freqwise)
                #     tps_preproc = time.time() - temp_start

                X_preproc = full_preprocessed_data(data_path,participants[j])
                if clf_name in ['PTGREEN','PTCNN']:
                    X_train, Y_train, X_test, Y_test, labels_code_test = get_train_test_data(Xt_preproc, Y[i], X_preproc,
                                                                                Y[j], domains[i], domains[j], codes, 
                                                                                labels_code[i], method, clf_name, dl.n_class,2, window_size,
                                                                                dl.freqwise, test_size)
                    if clf_name=='PTGREEN':
                        model = get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                    elif clf_name=='PTCNN':
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs) 
                    # perform the train test with Xsource and Xtarget
                    print("Train and test")
                    Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc = perform_measure_TF(model, X_train, Y_train, X_test, Y_test,
                                                                                                                    codes, dl.n_class,2, window_size,
                                                                                                                    dl.freqwise,test_size,batchsize,lr,
                                                                                                                    num_epochs)
                # Create the classifier
                if clf_name in ["TS_LDA","TS_SVM","MDM","CCNN","CNN","CGREEN","GREEN","DACNN"]:
                    X_train, Y_train, X_test, Y_test, labels_code_test = get_train_test_data(X[i], Y[i], X[j],
                                                                                Y[j], domains[i], domains[j], codes, 
                                                                                labels_code[i], method, clf_name, dl.n_class,2, window_size,
                                                                                dl.freqwise, test_size)

                    X_std = X_train.std(axis=0)
                    X_train = X_train/(X_std + 1e-8)
                    X_test = X_test/(X_std + 1e-8)
                    if clf_name=='CGREEN':
                        model = make_pipeline(
                                XdawnST(nfilter=4,classes=[1],estimator='lwf'),
                                Aligner(estimator="lwf",metric="real"),
                                get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                                )
                    if clf_name=='GREEN':
                        model = get_green(
                                    n_freqs=22,
                                    kernel_width_s=window_size,
                                    n_ch=8,
                                    sfreq=500,
                                    oct_min=0,
                                    oct_max=4.4,
                                    orth_weights=False,
                                    dropout=.6,
                                    hidden_dim=[20,10],
                                    logref='logeuclid',
                                    pool_layer=RealCovariance(),
                                    bi_out=[4],
                                    dtype=torch.float32,
                                    out_dim=2,
                                    )
                    if clf_name=="TS_LDA":
                        model = make_pipeline(XdawnCovariances(nfilter=4,xdawn_estimator="lwf",estimator="lwf",classes=[1]),
                            TangentSpace(), LDA(solver="lsqr", shrinkage="auto"))
                    if clf_name=="TS_SVM":
                        model = make_pipeline(XdawnCovariances(nfilter=4,xdawn_estimator="lwf",estimator="lwf",classes=[1]),
                            TangentSpace(), SVC())
                    elif clf_name=="MDM":
                        model = make_pipeline(XdawnCovariances(nfilter=4,xdawn_estimator="lwf",estimator="lwf",classes=[1]),MDM())
                    elif clf_name=="CNN":
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs)
                    elif clf_name=="CCNN":
                        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
                        model = make_pipeline(
                                XdawnST(nfilter=4,classes=[1],estimator='lwf'),
                                Aligner(estimator="lwf",metric="real"),EEG2Code(windows_size = X_train[0].shape[-1],
                                         n_channel_input = X_train[0].shape[-2],
                                         optimizer=optimizer,
                                         num_epochs=num_epochs)
                        )

                    # perform the train test with Xsource and Xtarget
                    print("Train and test")
                    Y_test, Y_pred, labels_pred_accumul, tps_train, tps_pred, tps_acc = perform_measure_TF(model, X_train, Y_train, X_test, Y_test,
                                                                                                                    codes, dl.n_class,2, window_size,
                                                                                                                    dl.freqwise,test_size,batchsize,lr,
                                                                                                                    num_epochs) 
                        
                # perform the measure of similarity between Xsource and Xtarget
                if method !="SiSu":
                    print("get the similarity score")
                    # similarity[participants[i]][participants[j]] = similarity_score(Xt_preproc, Y[i], X_preproc, Y[j])
            
                # Calcul the different classification metric 
                score, recall, f1, score_code = get_all_metrics(Y_test, Y_pred, labels_code_test, labels_pred_accumul)
                print("score_code",score_code)
                all_metrics[:,i,j] = np.array([tps_preproc, tps_train, tps_pred, tps_acc, score, recall, f1, score_code])

    save_path = '/'.join([file_path,"ws"+str(window_size),timewise,"results","cal_2",''])
    # np.save(save_path+"similarity_"+prefix+".npy",similarity)
    name = ["tps_preproc_"+prefix+".npy", "tps_train_"+prefix+".npy", "tps_pred_"+prefix+".npy", "tps_acc_"+prefix+".npy", "score_"+prefix+".npy","recall_"+prefix+".npy", "f1_"+prefix+".npy", "score_code_"+prefix+".npy"]
    for i in range(all_metrics.shape[0]):
        # np.save(save_path+name[i],all_metrics[i])
        print(i,all_metrics[i])

    return similarity, all_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants",default="['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24']", help="The index of the subject to get data from")
    parser.add_argument("--timewise",default="time_sample", help="The index of the subject to test on")
    parser.add_argument("--clf_name",default="1",help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--ws",default=0.35,type=float,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--nb_epoch",default=20,type=int,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--path",default='D:/s.velut/Documents/Thèse/Protheus_PHD/Data/Dry_Ricker/',help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--fpath",default='D:/s.velut/Documents/Thèse/Protheus_PHD/Data/STL',help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--method",default='DA',help="Boolean to recenter the data before classifying or not")

    dic_name = {"1":"GREEN","2":"TS_LDA","3":"MDM","5":"TS_SVM","6":"CNN"}

    path = 'D:/s.velut/Documents/Thèse/Protheus_PHD/Data/Dry_Ricker/'
    file_path = 'D:/s.velut/Documents/Thèse/Protheus_PHD/Data/STL'
    n_class=5
    fmin = 1
    fmax = 45
    fps = 60
    window_size = 0.35
    sfreq = 500
    num_epochs = 20
    timewise="time_sample"
    clf_name = "PTGREEN"
    method = "SiSu"
    participants = '["P1"]'
    # participants = '["P1","P17","P16","P14","P19","P23"]'
    # participants = "['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',\
    #                 'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',\
    #                 'P21','P22','P23','P24']"
    test_size=0.2
    batchsize=64
    lr=1e-03

    args = parser.parse_args()
    prefix = clf_name+method

    sim, metric = main(path,file_path,fmin,fmax,sfreq,fps,timewise,participants,clf_name,
                       method,window_size,test_size,batchsize,lr,num_epochs=num_epochs,prefix=prefix)
                       