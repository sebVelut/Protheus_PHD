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
import optuna
from optuna.trial import TrialState

def full_preprocessed_data(path,participant):
    return np.load(path+"full_preprocess_data_"+participant+".npy")


def solo_preprocessed_data(path,participant):
    return np.load(path+"full_solo_preprocess_data_"+participant+".npy")

def get_data():
    data_path = '/'.join(['D:/s.velut/Documents/Thèse/Protheus_PHD/Data/STL',"ws"+str(0.35),"time_sample","data",''])
    # get the data
    dl = STLDataLoader('D:/s.velut/Documents/Thèse/Protheus_PHD/Data/Dry_Ricker/', 1, 45, 0.35, 500, 60, "time_sample", ["P18"], 5)
    raw_data = dl.load_data()
    X, Yt, domains, codes, labels_code = dl.get_epochs(raw_data)
    Xt = solo_preprocessed_data(data_path,"P18")

    # Get the training data
    nb_sample_cal = int(5*7*(2.2-0.35)*500)   
    X_train = Xt[:nb_sample_cal]
    Y_train = Yt[0][:nb_sample_cal]
    X_test = Xt[nb_sample_cal:]
    Y_test = Yt[0][nb_sample_cal:]

    X_train, Y_train, domains_train = balance(X_train,Y_train,domains[0][:nb_sample_cal])

    return X_train, Y_train, X_test, Y_test

def objective(trial):
    # Generate the model.
    # model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    n_freqs = trial.suggest_int("n_freqs",6,30,step=2)
    dropout = trial.suggest_float("dropout",0.5,0.8,step=0.1)
    oct_max = trial.suggest_float("oct_max",4.0,5.0,step=0.2)
    clf = get_green(
                    n_freqs=n_freqs,
                    kernel_width_s=0.35,
                    n_ch=8,
                    sfreq=500,
                    oct_min=0,
                    oct_max=oct_max,
                    orth_weights=False,
                    dropout=dropout,
                    hidden_dim=[20,10],
                    logref='logeuclid',
                    pool_layer=RealCovariance(),
                    bi_out=[4],
                    dtype=torch.float32,
                    out_dim=2,
                    ) 

    X_train, Y_train, X_test, Y_test = get_data()
    print(X_train.shape,Y_train.shape)

    clf = clf.fit(np.array(X_train), Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = balanced_accuracy_score(Y_test,Y_pred)
    # # Get the FashionMNIST dataset.
    # train_loader, valid_loader = get_mnist()

    # # Training of the model.
    # for epoch in range(EPOCHS):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         # Limiting training data for faster epochs.
    #         if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
    #             break

    #         data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = F.nll_loss(output, target)
    #         loss.backward()
    #         optimizer.step()

    #     # Validation of the model.
    #     model.eval()
    #     correct = 0
    #     with torch.no_grad():
    #         for batch_idx, (data, target) in enumerate(valid_loader):
    #             # Limiting validation data.
    #             if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
    #                 break
    #             data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
    #             output = model(data)
    #             # Get the index of the max log-probability.
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        # trial.report(accuracy, epoch)

        # # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return accuracy

def main():

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

main()