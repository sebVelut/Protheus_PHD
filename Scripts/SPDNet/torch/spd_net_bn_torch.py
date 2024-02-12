
import torch.nn as nn
import torch
from torch.nn import Flatten
from skorch import NeuralNetClassifier

from .layers import BiMap, LogEig, ReEig, BatchNormSPD


class SPDNetBN_Module(nn.Module):
    def __init__(self, n_classes=2, bimap_dims=[64, 32], bn_momentum=0.1):
        super(SPDNetBN_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        self.bn_momentum = bn_momentum

        layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims:
            layers.append(BiMap(1, 1, input_dim, output_dim))
            layers.append(BatchNormSPD(momentum=self.bn_momentum, n=output_dim))
            layers.append(ReEig())
            input_dim = output_dim
        layers.append(LogEig())
        layers.append(Flatten())
        lin_layer = nn.Linear(self.bimap_dims[-1] ** 2, self.n_classes, bias=False,dtype=torch.float64)#.double()
        nn.init.xavier_uniform_(lin_layer.weight)
        layers.append(lin_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, None, :, :]
        y = self.net(x)
        return y


class SPDNetBN_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(SPDNetBN_Torch, self).__init__(**kwargs)