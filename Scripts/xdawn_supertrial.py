from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.utils.utils import check_function
from pyriemann.utils.covariance import normalize, get_nondiag_weight, cov_est_functions
import numpy as np
from scipy.linalg import eigh


class XdawnST(BaseEstimator, TransformerMixin):
    """Xdawn algorithm.

    Xdawn [1]_ is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originaly
    designed for P300 evoked potential by enhancing the target response with
    respect to the non-target response [2]_. This implementation is a
    generalization to any type of ERP.

    Parameters
    ----------
    nfilter : int, default=4
        The number of components to decompose M/EEG signals.
    classes : list of int | None, default=None
        List of classes to take into account for Xdawn.
        If None, all classes will be accounted.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    baseline_cov : None | array, shape(n_channels, n_channels), default=None
        Covariance matrix to which the average signals are compared. If None,
        the baseline covariance is computed across all trials and time samples.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    filters_ : ndarray, shape (n_classes x min(n_channels, n_filters), \
            n_channels)
        If fit, the Xdawn components used to decompose the data for each event
        type, concatenated.
    patterns_ : ndarray, shape (n_classes x min(n_channels, n_filters), \
            n_channels)
        If fit, the Xdawn patterns used to restore M/EEG signals for each event
        type, concatenated.
    evokeds_ : ndarray, shape (n_classes x min(n_channels, n_filters), n_times)
        If fit, the evoked response for each event type, concatenated.

    See Also
    --------
    XdawnCovariances

    References
    ----------
    .. [1] `xDAWN algorithm to enhance evoked potentials: application to
        brain-computer interface
        <https://hal.archives-ouvertes.fr/hal-00454568/fr/>`_
        B. Rivet, A. Souloumiac, V. Attina, and G. Gibert. IEEE Transactions on
        Biomedical Engineering, 2009, 56 (8), pp.2035-43.
    .. [2] `Theoretical analysis of xDAWN algorithm: application to an
        efficient sensor selection in a P300 BCI
        <https://hal.archives-ouvertes.fr/hal-00619997>`_
        B. Rivet, H. Cecotti, A. Souloumiac, E. Maby, J. Mattout. EUSIPCO 2011
        19th European Signal Processing Conference, Aug 2011, Barcelone, Spain.
        pp.1382-1386.
    """

    def __init__(self, nfilter=4, classes=None, estimator='scm',
                 baseline_cov=None):
        """Init."""
        self.nfilter = nfilter
        self.classes = classes
        self.estimator = estimator
        self.baseline_cov = baseline_cov

    @property
    def estimator_fn(self):
        return check_function(self.estimator, cov_est_functions)

    def fit(self, X, y):
        """Train Xdawn spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of trials.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        n_trials, n_channels, n_times = X.shape

        self.classes_ = (np.unique(y) if self.classes is None else
                         self.classes)

        Cx = self.baseline_cov
        if Cx is None:
            tmp = X.transpose((1, 2, 0))
            Cx = np.asarray(self.estimator_fn(
                tmp.reshape(n_channels, n_times * n_trials)
            ))

        self.evokeds_ = []
        self.filters_ = []
        self.patterns_ = []
        for c in self.classes_:
            # Prototyped response for each class
            P = np.mean(X[y == c], axis=0)

            # Covariance matrix of the prototyper response & signal
            C = np.asarray(self.estimator_fn(P))

            # Spatial filters
            evals, evecs = eigh(C, Cx)
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)
            V = evecs
            A = np.linalg.pinv(V.T)
            # create the reduced prototyped response
            self.filters_.append(V[:, 0:self.nfilter].T)
            self.patterns_.append(A[:, 0:self.nfilter].T)
            self.evokeds_.append(V[:, 0:self.nfilter].T @ P)

        self.evokeds_ = np.concatenate(self.evokeds_, axis=0)
        self.filters_ = np.concatenate(self.filters_, axis=0)
        self.patterns_ = np.concatenate(self.patterns_, axis=0)
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of trials.

        Returns
        -------
        Xf : ndarray, shape (n_trials, n_classes x min(n_channels, n_filters),\
                n_times)
            Set of spatialy filtered trials.
        """
        X = self.filters_ @ X
        
        return np.hstack([X,np.tile(self.evokeds_[None,:,:],(X.shape[0],1,1))])