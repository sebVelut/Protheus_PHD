import copy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import *


def compute_covariances(data, estimator='lwf'):
    data = copy.deepcopy(data)

    assert len(data.shape) == 3

    est = Covariances(estimator=estimator)

    covmats = est.transform(data)

    return covmats

