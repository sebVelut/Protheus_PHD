import numpy as np
import copy

from pyriemann.utils.geodesic import geodesic_riemann
from scipy.linalg import sqrtm, inv
from numpy import iscomplexobj, real, any, isfinite

from .covariance import compute_covariances, mean_covariance


def compute_ref_riemann(data=None, mean=None, dtype='covmat'):
    data = copy.deepcopy(data)
    if dtype != 'covmat':
        covmats = compute_covariances(data, estimator='lwf')
        data = covmats

    if mean is None:
        mean = mean_covariance(data, metric='riemann')

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:

        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        r_ra = inv(sqrtm(mean))

        if iscomplexobj(r_ra):
            print("WARNING! Covariance matrix was not SPD somehow. " +
                  "Can be caused by running ICA-EOG rejection, if " +
                  "not, check data!!")
            r_ra = real(r_ra).astype(np.float64)
        elif not any(isfinite(r_ra)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_ra = mean

    return r_ra


def compute_riemannian_alignment(data, mean=None, dtype='covmat'):
    data = copy.deepcopy(data)

    r_ra = compute_ref_riemann(data=data, mean=mean, dtype=dtype)

    if dtype == 'covmat':
        result = np.matmul(r_ra, data)
        result = np.matmul(result, r_ra)
    else:
        result = np.matmul(r_ra, data)

    return result

