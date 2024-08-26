import sys
sys.path.append('Scripts/Wavelets/')
from Jade.jade.wavelet_layers import RickerWavelet,XdawnCov

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def test_wavelet_layers():

    wave_layer = RickerWavelet(init_freq=np.array([1,2,4,8,16])/500,
                               sfreq=500,
                               f_min=0.5,
                               f_max=30)
    
    xdcov_layer = XdawnCov(ini_n_filter=16,
                           estimator='lwf',
                           xdawn_estimator='lwf',
                           )
    X = np.random.rand(150,32,126)
    y = np.random.randint(0,2,(150))

    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    inputs,lab = [[i,labels] for i,labels in dataloader][0]
    print("inputs of size ", inputs.shape)
    print("labels of size ", lab.shape)
    out = wave_layer(inputs)
    print(out.shape)
    assert out.shape == (64,5,32,126)

    out = out.reshape(-1,5*32,126)

    xout = xdcov_layer(out,lab)
    print(xout.shape)
    assert xout.shape == (64,32,32)

test_wavelet_layers()