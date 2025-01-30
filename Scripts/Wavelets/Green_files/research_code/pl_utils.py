from copy import deepcopy

import geotorch
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append('D:/s.velut/Documents/Thèse/Protheus_PHD/Scripts')
from SPDNet.SPD_torch.optimizers import riemannian_adam as torch_riemannian_adam
from Wavelets.Green_files.green.spd_layers import BiMap
from Wavelets.Green_files.green.spd_layers import LogMap
from Wavelets.Green_files.green.spd_layers import Shrinkage
from Wavelets.Green_files.green.wavelet_layers import PW_PLV
from Wavelets.Green_files.green.wavelet_layers import CombinedPooling
from Wavelets.Green_files.green.wavelet_layers import CrossCovariance
from Wavelets.Green_files.green.wavelet_layers import CrossPW_PLV
from Wavelets.Green_files.green.wavelet_layers import RealCovariance
from Wavelets.Green_files.green.wavelet_layers import WaveletConv


def vectorize_upper(X: Tensor) -> Tensor:
    """Upper vectorisation of F SPD matrices with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters
    ----------
    X : Tensor
        (N) x F x C x C

    Returns
    -------
    Tensor
        (N) x (C (C + 1) / 2)
    """
    # Upper triangular
    d = X.shape[-1]
    triu_idx = torch.triu_indices(d, d, 1)
    if X.dim() == 4:  # batch
        X_out = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, :, triu_idx[0], triu_idx[1]] * np.sqrt(2)
        ], dim=-1)
        return X_out
    elif X.dim() == 3:  # single tensor
        return torch.cat([torch.diagonal(X, dim1=-2, dim2=-1),
                         X[triu_idx[0], triu_idx[1]] * np.sqrt(2)],
                         dim=-1)


def vectorize_upper_one(X: Tensor):
    """
    Upper vectorisation of a single SPD matrix with multiplication of
    off-diagonal terms by sqrt(2) to preserve norm.

    Parameters:
    -----------
    X : Tensor
        The covariance matrix of shape (N x P x P).

    Returns:
    --------
    X_vec : Tensor
        The vectorized covariance matrix of shape (N x P * (P + 1) / 2).
    """
    assert X.dim() == 3
    _, size, _ = X.shape
    triu_indices = torch.triu_indices(size, size, offset=1)
    if X.dim() == 3:  # batch of matrices
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[:, triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    elif X.dim() == 2:  # single matrix
        X_vec = torch.cat([
            torch.diagonal(X, dim1=-2, dim2=-1),
            X[triu_indices[0], triu_indices[1]] * np.sqrt(2)
        ], dim=-1)
    return X_vec


class Green(nn.Module):
    def __init__(self,
                 conv_layers: nn.Module,
                 pooling_layers: nn.Module,
                 spd_layers: nn.Module,
                 head: nn.Module,
                 proj: nn.Module,
                 device: torch.device
                 ):
        """
        Neural network model that processes EEG epochs using convolutional
        layers, follwed by the computation of SPD features.

        Parameters:
        -----------
        conv_layers : nn.Module
            The convolutional layers that operate on the raw EEG signals.
        pooling_layers : nn.Module
            The pooling layers that convert the the convolved signals
            to SPD (Symmetric Positive Definite) features.
        spd_layers : nn.Module
            The SPD layers that operate on the SPD features.
        head : nn.Module
            The head layer that acts in the Euclidean space.
        proj : nn.Module
            The projection layer that projects the SPD features to the
            Euclidean space.
        device: torch.device
            The device to send the tensors
        """
        super(Green, self).__init__()
        self.conv_layers = conv_layers
        self.pooling_layers = pooling_layers
        self.spd_layers = spd_layers
        self.proj = proj
        self.head = head
        self.device = device

    def forward(self, X: Tensor):
        """
        Parameters
        ----------
        X : Tensor
            N x P x T
        """
        X_hat = self.conv_layers(X)
        X_hat = self.pooling_layers(X_hat)
        X_hat = self.spd_layers(X_hat)
        X_hat = self.proj(X_hat)
        # Vectorize the matrice depending on the wanted covariance
        if isinstance(
            self.pooling_layers, RealCovariance
        ) or isinstance(self.pooling_layers, CombinedPooling):
            X_hat = vectorize_upper(X_hat)

        elif isinstance(
            self.pooling_layers, CrossCovariance
        ) or isinstance(
            self.pooling_layers, CrossPW_PLV
        ):
            X_hat = vectorize_upper_one(X_hat)

        X_hat = torch.flatten(X_hat, start_dim=1)
        X_hat = self.head(X_hat)
        return X_hat
    
    def set_params(self):
        pass

    def fit(self,X_train,Y_train, epochs=20, batch_size=64,shuffle=True):
        """
        fit the green classifier
        parameters:
        X_train: numpy array S x C x T, mandatory
            training EEG data
        X_train: numpy array S x 1, mandatory
            training labels
        epochs: int, optional
            number of epochs, optional
        X_train: int, optional
            size of the batch, optional
        """
        # Separate the training data in training and validation set
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

        # Convert data into PyTorch tensors
        X_train_tensor = torch.tensor(x_train, dtype=torch.float64, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=self.device)
        X_val_tensor = torch.tensor(x_val, dtype=torch.float64, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self.device)

        # Create DataLoader for train, validation, and test sets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch_riemannian_adam.RiemannianAdam(self.parameters(), lr=1e-3)

        # Train the model
        for epoch in range(epochs):
            running_loss = 0.0
            train_y_pred= []
            y_train = []
            self.train()
            for inputs, labels in train_dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                labels = labels.to('cpu')
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs, 1)
                train_y_pred.append(predicted.to('cpu'))
                y_train.append(labels)

                running_loss += loss.item()
            
            # Calcul of the balanced accuracy
            train_accuracy = balanced_accuracy_score(np.concatenate(y_train),np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(train_y_pred)]))
            
            # Validation
            self.eval()
            val_correct = 0
            val_y_pred = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    val_correct += (predicted == labels.to('cpu')).sum().item()
                    val_y_pred.append(predicted)


            val_accuracy = balanced_accuracy_score(y_val,np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(val_y_pred)]))
            print(f"Epoch {epoch+1} train Accuracy: {train_accuracy} ||  Validation Accuracy: {val_accuracy}")

            print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

        print("Training finished!")

        return self
    
    def predict(self,X, batchsize=64):
        """
        predict the label
        parameters:
        X: numpy array,
            EEG data to predict their labels
        batchsize: int, optional
            size of the batches, By default 64
        """
        # Transform the data in tensors
        X_test_tensor = torch.tensor(X, dtype=torch.float64, device=self.device)
        test_dataset = TensorDataset(X_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        self.eval()
        y_pred= []
        # predict
        with torch.no_grad():
            for inputs in test_dataloader:
                outputs = self(inputs[0])
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to('cpu')
                y_pred.append(predicted)
        
        # print("getting accuracy of participant ", i)
        test_y_pred = np.concatenate(y_pred)

        y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in test_y_pred])

        return y_pred_norm



def get_green(
    n_freqs: int = 15,
    kernel_width_s: int = 5,
    conv_stride: int = 5,
    oct_min: float = 0,
    oct_max: float = 5.5,
    random_f_init: bool = False,
    shrinkage_init: float = -3.,
    logref: str = 'logeuclid',
    dropout: float = .333,
    n_ch: int = 21,
    hidden_dim: int = 32,
    sfreq: int = 125,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
    pool_layer: nn.Module = RealCovariance(),
    bi_out: int = None,
    out_dim: int = 1,
    orth_weights=True
):
    """
    Helper function to get a Green model.

    Parameters
    ----------
    n_freqs : int, optional
        Number of main frequencies in the wavelet family, by default 15
    kernel_width_s : int, optional
        Width of the kernel in seconds for the wavelets, by default 5
    conv_stride : int, optional
        Stride of the convolution operation for the wavelets, by default 5
    oct_min : float, optional
        Minimum foi in octave, by default 0
    oct_max : float, optional
        Maximum foi in octave, by default 5.5
    random_f_init : bool, optional
        Whether to randomly initialize the foi, by default False
    shrinkage_init : float, optional
        Initial shrinkage value before applying sigmoid funcion, by default -3.
    logref : str, optional
        Reference matrix used for LogEig layer, by default 'logeuclid'
    dropout : float, optional
        Dropout rate for FC layers, by default .333
    n_ch : int, optional
        Number of channels, by default 21
    hidden_dim : int, optional
        Dimension of the hidden layer, if None no hidden layer, by default 32
    sfreq : int, optional
        Sampling frequency, by default 125
    dtype : torch.dtype, optional
        Data type of the tensors, by default torch.float32
    device : torch.device, optional
        Device of the tensors, by default torch.device("cpu")
    pool_layer : nn.Module, optional
        Pooling layer, by default RealCovariance()
    bi_out : int, optional
        Dimension of the output layer after BiMap, by default None
    out_dim : int, optional
        Dimension of the output layer, by default 1
    orth_weigths, bool, optionnal
        Usage of orthogonal weigths, by default True

    Returns
    -------
    Green
        The Green model
    """

    # Convolution
    cplx_dtype = torch.complex128 if (
        dtype == torch.float64) else torch.complex64
    if random_f_init:
        foi_init = np.random.uniform(oct_min, oct_max, size=n_freqs)
        fwhm_init = -np.random.uniform(oct_min - 1, oct_max - 1, size=n_freqs)
    else:
        foi_init = np.linspace(oct_min, oct_max, n_freqs)
        fwhm_init = -np.linspace(oct_min - 1, oct_max - 1, n_freqs)

    conv_layers = nn.Sequential(*[
        WaveletConv(
            kernel_width_s=kernel_width_s,
            sfreq=sfreq,
            foi_init=foi_init,
            fwhm_init=fwhm_init,
            stride=conv_stride,
            dtype=cplx_dtype,
            scaling='oct'
        )])

    if isinstance(
            pool_layer, RealCovariance
    ) or isinstance(
            pool_layer, PW_PLV):
        n_compo = n_ch
        feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    elif isinstance(pool_layer, CrossCovariance):
        n_compo = int(n_ch * n_freqs)
        feat_dim = int(n_compo * (n_compo + 1) / 2)
        n_freqs = None

    elif isinstance(pool_layer, CombinedPooling):
        pool_layer_0 = pool_layer.pooling_layers[0]
        if isinstance(
            pool_layer_0, RealCovariance
        ) or isinstance(
                pool_layer_0, PW_PLV):
            n_compo = n_ch
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) /
                           2) * len(pool_layer.pooling_layers)
            n_freqs = n_freqs * len(pool_layer.pooling_layers)

        elif isinstance(pool_layer_0, CrossCovariance
                        ) or isinstance(pool_layer_0, CrossPW_PLV):
            n_compo = int(n_ch * n_freqs)
            feat_dim = int(n_compo * (n_compo + 1) / 2) * \
                len(pool_layer.pooling_layers)
            n_freqs = len(pool_layer.pooling_layers)

    # pooling
    pool_layer = pool_layer

    # SPD layers
    if shrinkage_init is None:
        spd_layers_list = [nn.Identity()]
    else:
        spd_layers_list = [Shrinkage(n_freqs=n_freqs,
                                     size=n_compo,
                                     init_shrinkage=shrinkage_init,
                                     learnable=True
                                     )]
    if bi_out is not None:
        for bo in bi_out:
            bimap = BiMap(d_in=n_compo,
                          d_out=bo,
                          n_freqs=n_freqs)
            if orth_weights:
                geotorch.orthogonal(bimap, 'weight')
            spd_layers_list.append(bimap)

            n_compo = bo

        if n_freqs is None:
            feat_dim = int(n_compo * (n_compo + 1) / 2)
        else:
            feat_dim = int(n_freqs * n_compo * (n_compo + 1) / 2)

    spd_layers = nn.Sequential(*spd_layers_list)

    # Projection to tangent space
    proj = LogMap(size=n_compo,
                  n_freqs=n_freqs,
                  ref=logref,
                  momentum=0.9,
                  reg=1e-4)

    # Head
    if hidden_dim is None:
        head = torch.nn.Sequential(*[
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype,
                                 device=device),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype,
                            device=device),
        ])
    else:
        # add multiple FC layers
        sequential_list = []
        for hd in hidden_dim:
            sequential_list.extend([
                torch.nn.BatchNorm1d(feat_dim,
                                     dtype=dtype,
                                     device=device),
                torch.nn.Dropout(
                    p=dropout) if dropout is not None else nn.Identity(),
                torch.nn.Linear(feat_dim,
                                hd,
                                dtype=dtype,
                                device=device),
                torch.nn.GELU()
            ])
            feat_dim = hd
        sequential_list.extend([
            torch.nn.BatchNorm1d(feat_dim,
                                 dtype=dtype,
                                 device=device),
            torch.nn.Dropout(
                p=dropout) if dropout is not None else nn.Identity(),
            torch.nn.Linear(feat_dim,
                            out_dim,
                            dtype=dtype,
                            device=device)
        ])
        head = torch.nn.Sequential(*sequential_list)

    # Gather everything
    model = Green(
        conv_layers=conv_layers,
        pooling_layers=pool_layer,
        spd_layers=spd_layers,
        head=head,
        proj=proj,
        device=device
    )
    return model
