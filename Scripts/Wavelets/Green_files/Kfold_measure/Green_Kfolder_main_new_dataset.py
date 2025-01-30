import numpy as np
from EEG2CodeKeras import basearchi
from Green_Kfold_measure_new_dataset import Green_Kfolder_ND
from pyriemann.estimation import Xdawn
import argparse
import mne
import os

from utils import Euc2SPD
from Alignments.riemannian import compute_riemannian_alignment
from collections import OrderedDict

def to_window_old(data, labels,length,n_samples_windows,codes,window_size=0.25,normalise=True,sfreq=500,fps=60,n_channels=8):
    
    X = np.empty(shape=((length)*data.shape[0], n_channels, n_samples_windows))
    idx_taken = []
    y = np.empty(shape=((length)*data.shape[0]), dtype=int)
    count = 0
    for trial_nb, trial in enumerate(data):
        lab = labels[trial_nb]
        c = codes[lab]
        code_pos = 0
        for idx in range(length):
            X[count] = trial[:, idx:idx+n_samples_windows]
            if idx/fps >= (code_pos+1)/fps:
                code_pos += 1 
            y[count] = int(c[code_pos])
            count += 1
        
        for idx in range(length):
            # print('Xidx:', trial_nb*length+idx, "Tidxm:", idx, 'TidxM:', idx +
            #       n_samples_windows, 'Ltrial', trial[:, idx:idx+n_samples_windows].shape)
            idx_taken.append(trial_nb*length+idx)
    # if normalise:
    #     X_std = X.std(axis=0)
    #     X /= X_std + 1e-8
    y_pred = np.vstack((y,np.abs(1-y))).T
    y = np.array([1 if (y >= 0.5) else 0 for y in y_pred[:,0]])

    return X, y, np.array(idx_taken)

def onset_anno(onset_window,label_window,onset_code,nb_seq_min,nb_seq_max,code_freq,sfreq,win_size):
    assert(sfreq!=0)
    new_onset = []
    new_onset_0 = []
    current_code = 0
    onset_code = np.ceil(onset_code*code_freq/sfreq)
    nb_seq_min-=1
    onset_shift = onset_code[current_code+nb_seq_min]
    time_trial = (2.2-win_size)
    # onset_window = np.arange(0,time_trial*code_freq*(nb_seq_max-nb_seq_min)-1,1,dtype=int)
    for i,o in enumerate(onset_window):
        if label_window[i]==1:
            # print(i)
            if current_code==nb_seq_max-1-nb_seq_min:
                new_onset.append(o+onset_shift)
            else:
                if o+onset_shift >= onset_code[current_code+nb_seq_min]+time_trial*code_freq:
                    current_code+=1
                    onset_shift = onset_code[current_code+nb_seq_min]-time_trial*code_freq*current_code
                new_onset.append(o+onset_shift)
        else:
            if current_code==nb_seq_max-1-nb_seq_min:
                new_onset_0.append(o+onset_shift)
            else:
                if o+onset_shift >= onset_code[current_code+nb_seq_min]+time_trial*code_freq:
                    current_code+=1
                    onset_shift = onset_code[current_code+nb_seq_min]-time_trial*code_freq*current_code
                new_onset_0.append(o+onset_shift)
    
    # modified_onset_code = [onset_code[i]-time_trial*sfreq*i for i in range(nb_seq_min,nb_seq_max)]
    # new_onset_0 = np.concatenate([np.arange(onset_code[i],onset_code[i]+time_trial*sfreq,sfreq//60) for i in range(nb_seq_min,nb_seq_max)])
    new_onset_0 = np.array(list(filter(lambda i: i not in new_onset, new_onset_0)))
    # print(new_onset_0.shape)
    return np.array(new_onset)/code_freq, np.array(new_onset_0)/code_freq
            

def get_data(subjects,recenter,window_size):
    participants = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10',
                'P11','P12','P13','P14','P15','P16','P17','P18','P19','P20',
                'P21','P22','P23','P24']
    # path = 'C:/Users/s.velut/Documents/These/Protheus_PHD/Data/Dry_Ricker'
    path = '/mnt/beegfs/home/velut/Seb/ComparaisonAlgo/new_dataset/Data/Dry_Ricker'
    n_class=5
    sfreq = 500
    fps = 60

    raw_eeglab = [mne.io.read_raw_eeglab(os.path.join(path, '_'.join([participants[i], 'dryburst100.set'])), preload=True, verbose=False)
               for i in range(len(participants))]

    for ind_i,i in enumerate(participants):
        raw_eeglab[ind_i] = raw_eeglab[ind_i].filter(l_freq=1, h_freq=25, method="fir", verbose=True)
    
    epochs_list = []
    events_list = []
    events_id_list = []
    onset_code_list = []
    data_list = []
    labels_code_list = []
    for ind_i, i in enumerate(participants): 
        # Strip the annotations that were script to make them easier to process
        events, event_id = mne.events_from_annotations(raw_eeglab[ind_i], event_id='auto', verbose=False)
        to_remove = []
        for idx in range(len(raw_eeglab[ind_i].annotations.description)):
            if (('boundary' in raw_eeglab[ind_i].annotations.description[idx]) or
                ('BURST' in raw_eeglab[ind_i].annotations.description[idx])):
                to_remove.append(idx)

        to_remove = np.array(to_remove)
        if len(to_remove) > 0:
            raw_eeglab[ind_i].annotations.delete(to_remove)
        # Get the events
        temp_event,temp_event_id = mne.events_from_annotations(raw_eeglab[ind_i], event_id='auto', verbose=False)
        events_list.append(temp_event)
        events_id_list.append(temp_event_id)
        shift = 0.0
        # Epoch the data following event
        epochs_list.append(mne.Epochs(raw_eeglab[ind_i], events_list[ind_i], event_id=events_id_list[ind_i], tmin=shift, \
                    tmax=2.2+shift, baseline=(None, None), preload=False, verbose=False))
        # print(events_list[ind_i])

        labels_code_list.append(epochs_list[ind_i].events[..., -1])
        labels_code_list[ind_i] -= np.min(labels_code_list[ind_i])
        # print(epochs.events[..., -1])
        data_list.append(epochs_list[ind_i].get_data())
        info_ep = epochs_list[ind_i].info
        # print(epochs)
        onset_code_list.append(epochs_list[ind_i].events[..., 0])
    data_list = np.array(data_list)

    codes = OrderedDict()
    for k, v in events_id_list[0].items():
        code = k.split('_')[0]
        code = code.replace('.','').replace('2','')
        idx = k.split('_')[1]
        codes[v-1] = np.array(list(map(int, code)))

    n_samples_windows = int(window_size*sfreq)
    length = int((2.2-window_size)*fps)
    X = np.zeros((data_list.shape[0],length*data_list.shape[1],data_list.shape[2],n_samples_windows))
    Y = np.zeros((data_list.shape[0],length*data_list.shape[1]))
    idx_taken = np.zeros((data_list.shape[0],length*data_list.shape[1]))
    domains = []
    # print(Y.shape)
    # print(X.shape)
    for ind_i,i in enumerate(participants):
        X[ind_i],Y[ind_i],idx_taken[ind_i] = to_window_old(data_list[ind_i],labels_code_list[ind_i],length,n_samples_windows,codes,window_size=window_size)
        domains.append(["Source_sub_{}".format(ind_i+1),]*len(Y[ind_i]))
    domains = np.array(domains)

    for ind_i,i in enumerate(participants):
        onset,onset_0 = onset_anno(idx_taken[ind_i],Y[ind_i],onset_code_list[ind_i],1,n_class*15,60,500,window_size)
        anno = mne.Annotations(onset,0.001*sfreq//60,"1")
        anno.append(onset_0,0.001*sfreq//60,"0")

        raw_eeglab[ind_i] = raw_eeglab[ind_i].set_annotations(anno)  

    X = np.zeros((data_list.shape[0],length*data_list.shape[1],data_list.shape[2]*2,int(window_size*sfreq)))
    y = np.zeros((data_list.shape[0],length*data_list.shape[1]))
    for ind_i,i in enumerate(participants):
        events, event_id = mne.events_from_annotations(raw_eeglab[ind_i])
        # print(event_id)
        epochs = mne.Epochs(raw_eeglab[ind_i],events,event_id,tmin=0.0,tmax=window_size,baseline=(0,0))
        temp_X = epochs.get_data()[:,:,:-1]
        y[ind_i] = epochs.events[...,-1]-1

        xdawn = Xdawn(nfilter=8,classes=[1],estimator='lwf')
        temp_X = xdawn.fit_transform(temp_X,y[ind_i])
        temp_X = np.hstack([temp_X,np.tile(xdawn.evokeds_[None,:,:],(temp_X.shape[0],1,1))])
        if recenter:
            X[ind_i] = compute_riemannian_alignment(temp_X, mean=None, dtype='real')
        

    return X,y,domains,labels_code_list,codes



def main(subjects,subtest,recenter,window_size):
    if not os.path.exists("./results/score/"):
        os.makedirs("./results/score/")
    if not os.path.exists("./results/score_code/"):
        os.makedirs("./results/score_code/")
    if not os.path.exists("./results/tps_train/"):
        os.makedirs("./results/tps_train/")
    if not os.path.exists("./results/tps_test/"):
        os.makedirs("./results/tps_test/")

    subjects = eval(subjects)
    recenter = eval(recenter)
    on_frame = eval(on_frame)
    subtest = subtest-1
    print(recenter)
    # tf.debugging.set_log_device_placement(True)
    # print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    # on_frame = eval(on_frame)
    # recenter = eval(recenter)
    prefix = ""
    if recenter:
        prefix = "recentered"

    # keep = ["O1", "O2", "Oz", "P7", "P3", "P4", "P8", "Pz","stim_trial","stim_epoch"]
    keep = None
    kfold = 10

    X_parent,Y_parent,domains_parent,labels_codes,codes = get_data(subjects,tospd=True)
    

    # Green Kfolder
    print("Perform Grenn\n")
    kf = Green_Kfolder_ND(kfold)
    kf.perform_Kfold(X_parent,Y_parent,domains_parent,labels_codes,codes,subjects,subtest,prefix,5,window_size)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects",default='[1,2,3,4,5,6,7,8,9,10,11,12]', help="The index of the subject to get data from")
    parser.add_argument("--subtest",default=1,type=int, help="The index of the subject to test on")
    parser.add_argument("--recenter",default=True,type=bool,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--ws",default=0.25,type=float,help="Boolean to recenter the data before classifying or not")

    args = parser.parse_args()

    main(args.subjects,args.subtest,args.on_frame,args.recenter,args.ws)
