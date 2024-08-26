
from utils import prepare_data,get_BVEP_data

from sklearn.manifold import TSNE
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from mne.decoding import Vectorizer


import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from matplotlib import colors
from moabb.paradigms import CVEP

sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts\\SPDNet")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40

"""
Visualisation des données avant et le classifieur et entre les couches
"""


### Fonction pour visualiser

def visualise(X,n_samples,space="real"):
    if space=="cov":
        X = TangentSpace().fit_transform(X)
    tsne = TSNE(n_components=2,perplexity=int(n_samples*2/3),n_iter=500)
    return tsne.fit_transform(X)

def data2vis(X,y,n_samples,sep='label',labels_codes=None):
    to_visualize = []
    if sep=='label':
        for k,data in enumerate(X):
            ind_1 = np.random.choice(np.where(y[k]==1)[0],size=n_samples,replace=False)
            ind_0 = np.random.choice(np.where(y[k]==0)[0],size=n_samples,replace=False)

            data1 = np.array([data[i] for i in ind_1])
            data0 = np.array([data[i] for i in ind_0])
            to_visualize.append(data1)
            to_visualize.append(data0)

    elif sep=="code":
        assert labels_codes is not None, "You need to give the labels of the code"
        for k,data in enumerate(X):
            labels = np.repeat(labels_codes,117,axis=1)
            ind_0 = np.random.choice(np.where(labels[k]==0)[0],size=n_samples,replace=False)
            ind_1 = np.random.choice(np.where(labels[k]==1)[0],size=n_samples,replace=False)
            ind_2 = np.random.choice(np.where(labels[k]==2)[0],size=n_samples,replace=False)
            ind_3 = np.random.choice(np.where(labels[k]==3)[0],size=n_samples,replace=False)

            data3 = np.array([data[i] for i in ind_3])
            data2 = np.array([data[i] for i in ind_2])
            data1 = np.array([data[i] for i in ind_1])
            data0 = np.array([data[i] for i in ind_0])
            to_visualize.append(data3)
            to_visualize.append(data2)
            to_visualize.append(data1)
            to_visualize.append(data0)

    return np.concatenate(np.array(to_visualize))

def Visualisation_castillos(subjects,n_samples,file_name,recenter=True,on_frame=True,normalise=True,tospd=False,sep='label'):
    dataset_moabb = CasitllosBurstVEP100()
    paradigm = CVEP()
    raw = dataset_moabb.get_data()

    raw_data,labels,codes,labels_codes = get_BVEP_data(subjects,on_frame)
    X,Y,domains = prepare_data(subjects,raw_data,labels,on_frame,tospd,recenter,codes=codes)

    to_visualize = data2vis(X,Y,n_samples,sep,labels_codes)
    print(to_visualize.shape)

    print("visualisation")
    if tospd:
        visualised = visualise(to_visualize,len(X)*2*n_samples,space='cov')
    else:
        to_visualize = to_visualize.reshape(-1,X.shape[-1]*X.shape[-2])
        visualised = visualise(to_visualize,len(X)*2*n_samples,space='real')

    print("saving")
    if sep=="code":
        file_name+="_code"
    np.save("C:/Users/s.velut/Documents/These/Protheus_PHD/results/Visualisation/Castillos/{}.npy".format(file_name),visualised)



# subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
n_samples = 150


# Visualisation_castillos(subjects,n_samples,"predata_nrc",recenter=False,sep="code")
# Visualisation_castillos(subjects,n_samples,"predata_rc",recenter=True,sep="code")
# Visualisation_castillos(subjects,n_samples,"predata_nrc_riem",recenter=False,tospd=True,sep="code")
# Visualisation_castillos(subjects,n_samples,"predata_rc_riem",recenter=True,tospd=True,sep="code")


Visualisation_castillos([1],n_samples,"predata_nrc_1",recenter=False)
Visualisation_castillos([1],n_samples,"predata_rc_1",recenter=True)