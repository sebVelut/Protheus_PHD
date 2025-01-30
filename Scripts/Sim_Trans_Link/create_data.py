import copy
import mne
import sys

from sklearn.preprocessing import RobustScaler
sys.path.append('D:/s.velut/Documents/Thèse/Protheus_PHD/Scripts')
mne.set_log_level('ERROR')
import numpy as np
from pyriemann.utils.mean import mean_covariance
from pyriemann.estimation import Xdawn
from Alignments.covariance import compute_covariances
from Alignments.riemannian import compute_riemannian_alignment
from STLDataLoader import STLDataLoader



def load_data(path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, n_class=5):
    dl = STLDataLoader(path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, n_class)
    raw_data = dl.load_data()
    X, Y, domains, codes, labels_code = dl.get_epochs(raw_data)

    return X, Y

def Preprocess_Full(X, Y):
    X_preproc = X.copy()
    for i in range(X.shape[0]):
        xdawn = Xdawn(nfilter=4,classes=[1],estimator='lwf')
        X_std = X[i].std(axis=0)
        temp_X = X[i]/(X_std + 1e-8)
        xdawn = xdawn.fit(temp_X,Y[i])
        temp_X = xdawn.transform(temp_X)
        X_preproc[i] = np.hstack([temp_X,np.tile(xdawn.evokeds_[None,:,:],(temp_X.shape[0],1,1))])
        X_preproc[i] = compute_riemannian_alignment(X_preproc[i], mean=None, dtype='real')
        del(temp_X)

        

    return X_preproc

def Preprocess_Online(Xt, Yt, n_class=5, n_cal=4, window_size=0.35, freqwise=500):
    """
    the data in input comes from non preprocessed data 
    """

    nb_sample_cal = int(n_class*n_cal*(2.2-window_size)*freqwise)
    X = np.zeros((Xt.shape[0],Xt.shape[1],Xt.shape[2]))
    xdawn = Xdawn(nfilter=4,classes=[1],estimator='lwf')

    X_std = Xt[:nb_sample_cal].std(axis=0)
    temp_X = Xt/(X_std + 1e-8)
    xdawn = xdawn.fit(temp_X[:nb_sample_cal],Yt[:nb_sample_cal])
    temp_X_train = xdawn.transform(temp_X[:nb_sample_cal])
    temp_Xtest = xdawn.transform(temp_X[nb_sample_cal:])
    X[:nb_sample_cal] = np.hstack([temp_X_train,np.tile(xdawn.evokeds_[None,:,:],(temp_X_train.shape[0],1,1))])
    X[nb_sample_cal:] = np.hstack([temp_Xtest,np.tile(xdawn.evokeds_[None,:,:],(temp_Xtest.shape[0],1,1))])
    rmean = mean_covariance(compute_covariances(X[:nb_sample_cal], estimator='lwf'), metric='riemann')
    X = compute_riemannian_alignment(X, mean=rmean, dtype='real')
    
    return X

def main(path, save_path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, n_class=5):
    if timewise=="time_sample":
        freqwise=500
    elif timewise=="frame":
        freqwise=60

    X, Y = load_data(path, fmin, fmax, window_size, sample_freq, fps, timewise, participants, n_class)
    print(X.shape,Y.shape)
    # X_preproc = Preprocess_Full(X,Y)

    for i in range(X.shape[0]):
        X_solo = Preprocess_Online(X[i], Y[i], n_class=5, n_cal=2, window_size=window_size, freqwise=freqwise)

        name_preproc = '/'.join(["ws"+str(window_size),timewise,"data","cal_2","full_preprocess_data_"+participants[i]+".npy"])
        name_solo = '/'.join(["ws"+str(window_size),timewise,"data","cal_2","full_solo_preprocess_data_"+participants[i]+".npy"])

        # np.save(save_path+name_preproc,X_preproc[i])
        np.save(save_path+name_solo,X_solo)

path = 'D:/s.velut/Documents/Thèse/Protheus_PHD/Data/Dry_Ricker/'
save_path = 'D:/s.velut/Documents/Thèse/Protheus_PHD/Data/STL/'
n_class=5
fmin = 1
fmax = 45
fps = 60
window_size = 0.35
sfreq = 500
timewise="time_sample"
participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',
                'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',
                'P21','P22','P23','P24']


main(path,save_path,fmin,fmax,window_size,sfreq,fps,timewise,participants)