from cmath import log
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Function as F


import math
from typing import Callable, Tuple
from typing import Any
from torch.autograd import Function, gradcheck
from torch.functional import Tensor
from torch.types import Number

# define the epsilon precision depending on the tensor datatype
EPS = {th.float32: 1e-4, th.float64: 1e-7}


class StiefelParameter(nn.Parameter):
    """ Parameter constrained to the Stiefel manifold (for BiMap layers) """
    pass


def init_bimap_parameter(W):
    """ initializes a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            v = th.empty(ni, ni, dtype=W.dtype, device=W.device).uniform_(0., 1.)
            vv = th.svd(v.matmul(v.t()))[0][:, :no]
            W.data[i, j] = vv


def init_bimap_parameter_identity(W):
    """ initializes to identity a (ho,hi,ni,no) 4D-StiefelParameter"""
    ho, hi, ni, no = W.shape
    for i in range(ho):
        for j in range(hi):
            W.data[i, j] = th.eye(ni, no)


class SPDParameter(nn.Parameter):
    """ Parameter constrained to the SPD manifold (for ParNorm) """
    pass


def bimap(X, W):
    '''
    Bilinear mapping function
    :param X: Input matrix of shape (batch_size,n_in,n_in)
    :param W: Stiefel parameter of shape (n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,n_out,n_out)
    '''
    return W.t().matmul(X).matmul(W)


def bimap_channels(X, W):
    '''
    Bilinear mapping function over multiple input and output channels
    :param X: Input matrix of shape (batch_size,channels_in,n_in,n_in)
    :param W: Stiefel parameter of shape (channels_out,channels_in,n_in,n_out)
    :return: Bilinearly mapped matrix of shape (batch_size,channels_out,n_out,n_out)
    '''
    # Pi=th.zeros(X.shape[0],1,W.shape[-1],W.shape[-1],dtype=X.dtype,device=X.device)
    # for j in range(X.shape[1]):
    #     Pi=Pi+bimap(X,W[j])
    batch_size, channels_in, n_in, _ = X.shape
    channels_out, _, _, n_out = W.shape
    P = th.zeros(batch_size, channels_out, n_out, n_out, dtype=X.dtype, device=X.device)
    for co in range(channels_out):
        P[:, co, :, :] = sum([bimap(X[:, ci, :, :], W[co, ci, :, :]) for ci in range(channels_in)])
    return P


def modeig_forward(P, op, eig_mode='svd', param=None):
    '''
    Generic forward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    batch_size, channels, n, n = P.shape  # batch size,channel depth,dimension
    U, S = th.zeros_like(P, device=P.device), th.zeros(batch_size, channels, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            # if (eig_mode == 'eig'):
            #     s, U[i, j] = th.linalg.eig(P[i, j])
            #     S[i, j] = s[:, 0]
            # elif (eig_mode == 'svd'):
            U[i, j], S[i, j], _ = th.svd(P[i, j])
    S_fn = op.fn(S, param)
    X = U.matmul(BatchDiag(S_fn)).matmul(U.transpose(2, 3))
    return X, U, S, S_fn


def modeig_backward(dx, U, S, S_fn, op, param=None):
    '''
    Generic backward function of non-linear eigenvalue modification
    LogEig, ReEig, etc inherit from this class
    Input P: (batch_size,channels) SPD matrices of size (n,n)
    Output X: (batch_size,channels) modified symmetric matrices of size (n,n)
    '''
    # if __debug__:
    #     import pydevd
    #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
    S_fn_deriv = BatchDiag(op.fn_deriv(S, param))
    SS = S[..., None].repeat(1, 1, 1, S.shape[-1])
    SS_fn = S_fn[..., None].repeat(1, 1, 1, S_fn.shape[-1])
    L = (SS_fn - SS_fn.transpose(2, 3)) / (SS - SS.transpose(2, 3))
    L[L == -np.inf] = 0;
    L[L == np.inf] = 0;
    L[th.isnan(L)] = 0
    L = L + S_fn_deriv
    dp = L * (U.transpose(2, 3).matmul(dx).matmul(U))
    dp = U.matmul(dp).matmul(U.transpose(2, 3))
    return dp


class LogEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Log_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Log_op)


class ReEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Re_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Re_op)


class ExpEig(F):
    """
    Input P: (batch_size,h) symmetric matrices of size (n,n)
    Output X: (batch_size,h) of exponential eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Exp_op, eig_mode='eig')
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Exp_op)


class SqmEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqm_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqm_op)


class SqminvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse square root eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Sqminv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Sqminv_op)


class PowerEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of power eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P, power):
        Power_op._power = power
        X, U, S, S_fn = modeig_forward(P, Power_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Power_op), None


class InvEig(F):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of inverse eigenvalues matrices of size (n,n)
    """

    @staticmethod
    def forward(ctx, P):
        X, U, S, S_fn = modeig_forward(P, Inv_op)
        ctx.save_for_backward(U, S, S_fn)
        return X

    @staticmethod
    def backward(ctx, dx):
        # if __debug__:
        #     import pydevd
        #     pydevd.settrace(suspend=False, trace_only_current_thread=True)
        U, S, S_fn = ctx.saved_variables
        return modeig_backward(dx, U, S, S_fn, Inv_op)


def geodesic(A, B, t):
    '''
    Geodesic from A to B at step t
    :param A: SPD matrix (n,n) to start from
    :param B: SPD matrix (n,n) to end at
    :param t: scalar parameter of the geodesic (not constrained to [0,1])
    :return: SPD matrix (n,n) along the geodesic
    '''
    M = CongrG(PowerEig.apply(CongrG(B, A, 'neg'), t), A, 'pos')[0, 0]
    return M


def cov_pool(f, reg_mode='mle'):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    bs, n, T = f.shape
    X = f.matmul(f.transpose(-1, -2)) / (T - 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(X, 0.75)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def cov_pool_mu(f, reg_mode):
    """
    Input f: Temporal n-dimensionnal feature map of length T (T=1 for a unitary signal) (batch_size,n,T)
    Output ret: Covariance matrix of size (batch_size,1,n,n)
    """
    alpha = 1
    bs, n, T = f.shape
    mu = f.mean(-1, True);
    f = f - mu
    X = f.matmul(f.transpose(-1, -2)) / (T - 1) + alpha * mu.matmul(mu.transpose(-1, -2))
    aug1 = th.cat((X, alpha * mu), 2)
    aug2 = th.cat((alpha * mu.transpose(1, 2), th.ones(mu.shape[0], 1, 1, dtype=mu.dtype, device=f.device)), 2)
    X = th.cat((aug1, aug2), 1)
    if (reg_mode == 'mle'):
        ret = X
    elif (reg_mode == 'add_id'):
        ret = add_id(X, 1e-6)
    elif (reg_mode == 'adjust_eig'):
        ret = adjust_eig(0.75)(X)
    if (len(ret.shape) == 3):
        return ret[:, None, :, :]
    return ret


def add_id(P, alpha):
    '''
    Input P of shape (batch_size,1,n,n)
    Add Id
    '''
    for i in range(P.shape[0]):
        P[i] = P[i] + alpha * P[i].trace() * th.eye(P[i].shape[-1], dtype=P.dtype, device=P.device)
    return P


def dist_riemann(x, y):
    '''
    Riemannian distance between SPD matrices x and SPD matrix y
    :param x: batch of SPD matrices (batch_size,1,n,n)
    :param y: single SPD matrix (n,n)
    :return:
    '''
    return LogEig.apply(CongrG(x, y, 'neg')).view(x.shape[0], x.shape[1], -1).norm(p=2, dim=-1)


def CongrG(P, G, mode):
    """
    Input P: (batch_size,channels) SPD matrices of size (n,n) or single matrix (n,n)
    Input G: matrix (n,n) to do the congruence by
    Output PP: (batch_size,channels) of congruence by sqm(G) or sqminv(G) or single matrix (n,n)
    """
    if (mode == 'pos'):
        GG = SqmEig.apply(G[None, None, :, :])
    elif (mode == 'neg'):
        GG = SqminvEig.apply(G[None, None, :, :])
    PP = GG.matmul(P).matmul(GG)
    return PP


def LogG(x, X):
    """ Logarithmc mapping of x on the SPD manifold at X """
    return CongrG(LogEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def ExpG(x, X):
    """ Exponential mapping of x on the SPD manifold at X """
    return CongrG(ExpEig.apply(CongrG(x, X, 'neg')), X, 'pos')


def BatchDiag(P):
    """
    Input P: (batch_size,channels) vectors of size (n)
    Output Q: (batch_size,channels) diagonal matrices of size (n,n)
    """
    batch_size, channels, n = P.shape  # batch size,channel depth,dimension
    Q = th.zeros(batch_size, channels, n, n, dtype=P.dtype, device=P.device)
    for i in range(batch_size):
        for j in range(channels):
            Q[i, j] = P[i, j].diag()
    return Q


def karcher_step(x, G, alpha):
    '''
    One step in the Karcher flow
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mean(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def BaryGeom(x):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    with th.no_grad():
        G = th.mean(x, dim=0)[0, :, :]
        for _ in range(k):
            G = karcher_step(x, G, alpha)
        return G


def karcher_step_weighted(x, G, alpha, weights):
    '''
    One step in the Karcher flow
    Weights is a weight vector of shape (batch_size,)
    Output is mean of shape (n,n)
    '''
    x_log = LogG(x, G)
    G_tan = x_log.mul(weights[:, None, None, None]).sum(dim=0)[None, ...]
    G = ExpG(alpha * G_tan, G)[0, 0]
    return G


def bary_geom_weighted(x, weights):
    '''
    Function which computes the weighted Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Weights is a weight vector of shape (batch_size,)
    Output is (1,1,n,n) Riemannian mean
    '''
    k = 1
    alpha = 1
    # with th.no_grad():
    G = x.mul(weights[:, None, None, None]).sum(dim=0)[0, :, :]
    for _ in range(k):
        G = karcher_step_weighted(x, G, alpha, weights)
    return G[None, None, :, :]


class Log_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.log(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 1 / S


class Re_op():
    """ Log function and its derivative """
    _threshold = 1e-4

    @classmethod
    def fn(cls, S, param=None):
        return nn.Threshold(cls._threshold, cls._threshold)(S)

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (S > cls._threshold).double()


class Sqm_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return 0.5 / th.sqrt(S)


class Sqminv_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return 1 / th.sqrt(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return -0.5 / th.sqrt(S) ** 3


class Power_op():
    """ Power function and its derivative """
    _power = 1

    @classmethod
    def fn(cls, S, param=None):
        return S ** cls._power

    @classmethod
    def fn_deriv(cls, S, param=None):
        return (cls._power) * S ** (cls._power - 1)


class Inv_op():
    """ Inverse function and its derivative """

    @classmethod
    def fn(cls, S, param=None):
        return 1 / S

    @classmethod
    def fn_deriv(cls, S, param=None):
        return log(S)


class Exp_op():
    """ Log function and its derivative """

    @staticmethod
    def fn(S, param=None):
        return th.exp(S)

    @staticmethod
    def fn_deriv(S, param=None):
        return th.exp(S)


def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


def broadcast_dims(A: th.Size, B: th.Size, raise_error:bool=True) -> Tuple:
    """Return the dimensions that can be broadcasted.
    Parameters
    ----------
    A : torch.Size
        shape of first tensor
    B : torch.Size
        shape of second tensor
    raise_error : bool (=True)
        flag that indicates if an error should be raised if A and B cannot be broadcasted
    -------
    Returns : torch.Tensor
    """
    # check if the tensors can be broadcasted
    if raise_error:
        if len(A) != len(B):
            raise ValueError('The number of dimensions must be equal!')

    tdim = th.tensor((A, B), dtype=th.int32)

    # find differing dimensions
    bdims = tuple(th.where(tdim[0].ne(tdim[1]))[0].tolist())

    # check if one of the different dimensions has size 1
    if raise_error:
        if not tdim[:,bdims].eq(1).any(dim=0).all():
            raise ValueError('Broadcast not possible! One of the dimensions must be 1.')

    return bdims


def sum_bcastdims(A: Tensor, shape_out: th.Size) -> Tensor:
    """Returns a tensor whose values along the broadcast dimensions are summed.
    Parameters
    ----------
    A : torch.Tensor
        tensor that should be modified
    shape_out : torch.Size
        desired shape of the tensor after aggregation
    -------
    Returns : the aggregated tensor with the desired shape
    """
    bdims = broadcast_dims(A.shape, shape_out)

    if len(bdims) == 0:
        return A
    else:
        return A.sum(dim=bdims, keepdim=True)


def randn_sym(shape, **kwargs):
    ndim = shape[-1]
    X = th.randn(shape, **kwargs)
    ixs = th.tril_indices(ndim,ndim, offset=-1)
    X[...,ixs[0],ixs[1]] /= math.sqrt(2)
    X[...,ixs[1],ixs[0]] = X[...,ixs[0],ixs[1]]
    return X


def spd_2point_interpolation(A : Tensor, B : Tensor, t : Number) -> Tensor:
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    return rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, th.tensor(t)) @ rm_sq


class reverse_gradient(Function):
    """
    Reversal of the gradient 
    Parameters
    ---------
    scaling : Number 
        A constant number that is multiplied to the sign-reversed gradients (1.0 default)
    """
    @staticmethod
    def forward(ctx, x, scaling = 1.0):
        ctx.scaling = scaling
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.scaling
        return grad_output, None


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False) -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function
        ensure_symmetric : bool = False (optional)
            if ensure_symmetric=True, then M is symmetrized
        ensure_psd : bool = False (optional)
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        if ensure_symmetric:
            M = ensure_sym(M)

        # compute the eigenvalues and vectors
        s, U = th.linalg.eigh(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ th.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor, 
                 fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:
        """Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_reeig(Function):
    """
    Rectifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).
    """
    @staticmethod
    def value(s : Tensor, threshold : Tensor) -> Tensor:
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s : Tensor, threshold : Tensor) -> Tensor:
        return (s>threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold), None, None

    @staticmethod
    def tests():
        """
        Basic unit tests and test to check gradients
        """
        ndim = 2
        nb = 1
        # generate random base SPD matrix
        A = th.randn((1,ndim,ndim), dtype=th.double)
        U, s, _ = th.linalg.svd(A)

        threshold = th.tensor([1e-3], dtype=th.double)

        # generate batches
        # linear case (all eigenvalues are above the threshold)
        s = threshold * 1e1 + th.rand((nb,ndim), dtype=th.double) * threshold
        M = U @ th.diag_embed(s) @ U.transpose(-1,-2)

        assert (sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True))) 

        # non-linear case (some eigenvalues are below the threshold)
        s = th.rand((nb,ndim), dtype=th.double) * threshold
        s[::2] += threshold 
        M = U @ th.diag_embed(s) @ U.transpose(-1,-2)
        assert (~sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))

        # linear case, all eigenvalues are identical
        s = th.ones((nb,ndim), dtype=th.double)
        M = U @ th.diag_embed(s) @ U.transpose(-1,-2)
        assert (sym_reeig.apply(M, threshold, True).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))


class sym_abseig(Function):
    """
    Computes the absolute values of all eigenvalues for a batch symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.abs()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.sign()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_abseig.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_abseig.derivative), None


class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative 
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """
    Computes the matrix exponential for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """
    Computes the matrix power for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, exponent : Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s : Tensor, exponent : Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent-1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        dXs = (U.transpose(-1,-2) @ ensure_sym(dX) @ U).diagonal(dim1=-1,dim2=-2)
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(Function):
    """
    Computes the matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(Function):
    """
    Computes the inverse matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(Function):
    """
    Computes the square root and inverse square root matrices for a batch of SPD matrices.
    """

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ th.diag_embed(smod2) @ U.transpose(-1,-2)
        ctx.save_for_backward(s, smod, smod2, U)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        return dMsq + dMinvsq, None


class sym_invm(Function):
    """
    Computes the inverse matrices for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).reciprocal()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -1. * s.pow(-2)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invm.derivative), None


def spd_mean_kracher_flow(X : Tensor, G0 : Tensor = None, maxiter : int = 50, dim = 0, weights = None, return_dist = False, return_XT = False) -> Tensor:

    if X.shape[dim] == 1:
        if return_dist:
            return X, th.tensor([0.0], dtype=X.dtype, device=X.device)
        else:
            return X

    if weights is None:
        n = X.shape[dim]
        weights = th.ones((*X.shape[:-2], 1, 1), dtype=X.dtype, device=X.device)
        weights /= n

    if G0 is None:
        G = (X * weights).sum(dim=dim, keepdim=True)
    else:
        G = G0.clone()

    nu = 1.
    dist = tau = crit = th.finfo(X.dtype).max
    i = 0

    while (crit > EPS[X.dtype]) and (i < maxiter) and (nu > EPS[X.dtype]):
        i += 1

        Gsq, Ginvsq = sym_invsqrtm2.apply(G)
        XT = sym_logm.apply(Ginvsq @ X @ Ginvsq)
        GT = (XT * weights).sum(dim=dim, keepdim=True)
        G = Gsq @ sym_expm.apply(nu * GT) @ Gsq

        if return_dist:
            dist = th.norm(XT - GT, p='fro', dim=(-2,-1))
        crit = th.norm(GT, p='fro', dim=(-2,-1)).max()
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    if return_dist:
        return G, dist
    if return_XT:
        return G, XT
    return G
