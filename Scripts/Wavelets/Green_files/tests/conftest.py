import mne
import numpy as np
import pytest


def make_one_dummy_epoch(n_epochs):
    info = mne.create_info(
        # ch_names=['Oz', 'Pz', 'Cz'],
        ch_names=[str(i) for i in range(32)],
        sfreq=500,
        ch_types='eeg'
    )
    # data = np.random.randn(2, 32, 126)
    data = np.random.randn(n_epochs, 32, 126)
    epoch = mne.EpochsArray(data, info)
    return epoch


@pytest.fixture
def dummy_mne_epochs():
    epochs = [make_one_dummy_epoch() for _ in range(10)]
    return epochs


@pytest.fixture
def dummy_spd_ill_cond():
    # N x F x C
    U = np.exp(np.random.randn(2, 4, 3))
    U[:, :, 0] = -1e-6
    diag = U[..., np.newaxis] * np.eye(3)
    V = np.random.randn(2, 4, 3, 3)
    spd_ill = V @ diag @ np.transpose(V, (0, 1, 3, 2))
    return spd_ill
