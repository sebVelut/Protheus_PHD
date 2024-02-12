
import torch.nn as nn
from torch.nn import Flatten
from skorch import NeuralNetClassifier

from .layers import BiMap, LogEig, ReEig


class SPDNet_Module(nn.Module):
    def __init__(self, n_classes=2, bimap_dims=[64, 32]):
        super(SPDNet_Module, self).__init__()

        self.n_classes = n_classes
        self.bimap_dims = bimap_dims
        layers = []
        input_dim = bimap_dims[0]
        for output_dim in self.bimap_dims:
            layers.append(BiMap(1, 1, input_dim, output_dim))
            layers.append(ReEig())
            input_dim = output_dim
        layers.append(LogEig())
        layers.append(Flatten())
        lin_layer = nn.Linear(self.bimap_dims[-1] ** 2, self.n_classes, bias=False).double()
        nn.init.xavier_uniform_(lin_layer.weight)
        layers.append(lin_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x[:, None, :, :]
        y = self.net(x)
        return y


class SPDNet_Torch(NeuralNetClassifier):

    def __init__(self, **kwargs):

        super(SPDNet_Torch, self).__init__(**kwargs)