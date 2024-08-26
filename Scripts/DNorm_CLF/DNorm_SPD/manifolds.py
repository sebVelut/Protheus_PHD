"""Manifold of the Euclidean space."""
import abc
import tensorflow as tf


class Manifold(metaclass=abc.ABCMeta):
    name = "Base"
    ndims = None

    def __repr__(self):
        """Returns a string representation of the particular manifold."""
        return "{} (ndims={}) manifold".format(self.name, self.ndims)

    def check_shape(self, shape_or_tensor):
        """Check if given shape is compatible with the manifold."""
        shape = (
            shape_or_tensor.shape
            if hasattr(shape_or_tensor, "shape")
            else shape_or_tensor
        )
        return (len(shape) >= self.ndims) & self._check_shape(shape)

    def check_point_on_manifold(self, x, atol=None, rtol=None):
        """Check if point :math:`x` lies on the manifold."""
        return self.check_shape(x) & self._check_point_on_manifold(
            x, atol=atol, rtol=rtol
        )

    def check_vector_on_tangent(self, x, u, atol=None, rtol=None):
        """Check if vector :math:`u` lies on the tangent space at :math:`x`."""
        return (
            self._check_point_on_manifold(x, atol=atol, rtol=rtol)
            & self.check_shape(u)
            & self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        )

    def _check_shape(self, shape):
        return tf.constant(True)

    @abc.abstractmethod
    def _check_point_on_manifold(self, x, atol, rtol):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(self, x, u, atol, rtol):
        raise NotImplementedError

    @abc.abstractmethod
    def dist(self, x, y, keepdims=False):
        """Compute the distance between two points :math:`x` and :math:`y: along a
        geodesic.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, x, u, v, keepdims=False):
        """Return the inner product (i.e., the Riemannian metric) between two tangent
        vectors :math:`u` and :math:`v` in the tangent space at :math:`x`.
        """
        raise NotImplementedError

    def norm(self, x, u, keepdims=False):
        """Compute the norm of a tangent vector :math:`u` in the tangent space at
        :math:`x`.
        """
        return self.inner(x, u, u, keepdims=keepdims) ** 0.5

    @abc.abstractmethod
    def proju(self, x, u):
        """Project a vector :math:`u` in the ambient space on the tangent space at
        :math:`x`
        """
        raise NotImplementedError

    def egrad2rgrad(self, x, u):
        """Map the Euclidean gradient :math:`u` in the ambient space on the tangent
        space at :math:`x`.
        """
        return self.proju(x, u)

    @abc.abstractmethod
    def projx(self, x):
        """Project a point :math:`x` on the manifold."""
        raise NotImplementedError

    @abc.abstractmethod
    def retr(self, x, u):
        """Perform a retraction from point :math:`x` with given direction :math:`u`."""
        raise NotImplementedError

    @abc.abstractmethod
    def exp(self, x, u):
        r"""Perform an exponential map :math:`\operatorname{Exp}_x(u)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def log(self, x, y):
        r"""Perform a logarithmic map :math:`\operatorname{Log}_{x}(y)`."""
        raise NotImplementedError

    @abc.abstractmethod
    def transp(self, x, y, v):
        r"""Perform a vector transport :math:`\mathfrak{T}_{x\to y}(v)`."""
        return NotImplementedError

    def ptransp(self, x, y, v):
        r"""Perform a parallel transport :math:`\operatorname{P}_{x\to y}(v)`."""
        return NotImplementedError

    def random(self, shape, dtype=tf.float32):
        """Sample a random point on the manifold."""
        return self.projx(tf.random.uniform(shape, dtype=dtype))

    def geodesic(self, x, u, t):
        """Geodesic from point :math:`x` in the direction of tanget vector
        :math:`u`
        """
        raise NotImplementedError

    def pairmean(self, x, y):
        """Compute a Riemannian (Fréchet) mean of points :math:`x` and :math:`y`"""
        return self.geodesic(x, self.log(x, y), 0.5)


class Euclidean(Manifold):
    name = "Euclidean"
    ndims = 0

    def __init__(self, ndims=0):
        """Instantiate the Euclidean manifold.

        Args:
          ndims: number of dimensions
        """
        super().__init__()
        self.ndims = ndims

    def _check_point_on_manifold(self, x, atol, rtol):
        return tf.constant(True)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        return tf.constant(True)

    def dist(self, x, y, keepdims=False):
        return self.norm(x, x - y, keepdims=keepdims)

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(
            u * v, axis=tuple(range(-self.ndims, 0)), keepdims=keepdims
        )

    def proju(self, x, u):
        return u

    def projx(self, x):
        return x

    def exp(self, x, u):
        return x + u

    retr = exp

    def log(self, x, y):
        return y - x

    def ptransp(self, x, y, v):
        return v

    transp = ptransp

    def geodesic(self, x, u, t):
        return x + t * u

    def pairmean(self, x, y):
        return (x + y) / 2.0


class _Stiefel(Manifold):
    """Manifold of orthonormal p-frames in the n-dimensional Euclidean space."""

    ndims = 2

    def _check_shape(self, shape):
        return shape[-2] >= shape[-1]

    def _check_point_on_manifold(self, x, atol, rtol):
        xtx = transposem(x) @ x
        eye = tf.eye(
            tf.shape(xtx)[-1], batch_shape=tf.shape(xtx)[:-2], dtype=xtx.dtype
        )
        return utils.allclose(xtx, eye, atol, rtol)

    def _check_vector_on_tangent(self, x, u, atol, rtol):
        diff = transposem(u) @ x + utils.transposem(x) @ u
        return utils.allclose(diff, tf.zeros_like(diff), atol, rtol)

    def projx(self, x):
        _s, u, vt = tf.linalg.svd(x)
        return u @ vt

class StiefelEuclidean(_Stiefel):
    """Manifold of orthonormal p-frames in the n-dimensional space endowed with
    the Euclidean inner product.

    """

    name = "Euclidean Stiefel"

    def inner(self, x, u, v, keepdims=False):
        return tf.reduce_sum(u * v, axis=[-2, -1], keepdims=keepdims)

    def proju(self, x, u):
        xtu = transposem(x) @ u
        xtu_sym = (transposem(xtu) + xtu) / 2.0
        return u - x @ xtu_sym

    def exp(self, x, u):
        return self.geodesic(x, u, 1.0)

    def retr(self, x, u):
        q, r = tf.linalg.qr(x + u)
        unflip = tf.cast(tf.sign(tf.linalg.diag_part(r)), r.dtype)
        return q * unflip[..., tf.newaxis, :]

    def geodesic(self, x, u, t):
        xtu = utils.transposem(x) @ u
        utu = utils.transposem(u) @ u
        eye = tf.eye(
            tf.shape(utu)[-1], batch_shape=tf.shape(utu)[:-2], dtype=x.dtype
        )
        logw = blockm(xtu, -utu, eye, xtu)
        w = tf.linalg.expm(t * logw)
        z = tf.concat([tf.linalg.expm(-xtu * t), tf.zeros_like(utu)], axis=-2)
        y = tf.concat([x, u], axis=-1) @ w @ z
        return y

    def dist(self, x, y, keepdims=False):
        raise NotImplementedError

    def log(self, x, y):
        return NotImplementedError

    def transp(self, x, y, v):
        return self.proju(y, v)


def assign_to_manifold(var, manifold):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if not manifold.check_shape(var):
        raise ValueError("Invalid variable shape {}".format(var.shape))
    setattr(var, "manifold", manifold)


def get_eps(val):
    return np.finfo(val.dtype.name).eps


def allclose(x, y, rtol=None, atol=None):
    """Return True if two arrays are element-wise equal within a tolerance."""
    rtol = 10 * get_eps(x) if rtol is None else rtol
    atol = 10 * get_eps(x) if atol is None else atol
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def logm(inp):
    """Compute the matrix logarithm of positive-definite real matrices."""
    inp = tf.convert_to_tensor(inp)
    complex_dtype = tf.complex128 if inp.dtype == tf.float64 else tf.complex64
    log = tf.linalg.logm(tf.cast(inp, dtype=complex_dtype))
    return tf.cast(log, dtype=inp.dtype)


def transposem(inp):
    """Transpose multiple matrices."""
    perm = list(range(len(inp.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return tf.transpose(inp, perm)


def get_manifold(var, default_manifold=Euclidean()):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if hasattr(var, "manifold"):
        return var.manifold
    else:
        return default_manifold


def get_eps(val):
    return np.finfo(val.dtype.name).eps


def allclose(x, y, rtol=None, atol=None):
    """Return True if two arrays are element-wise equal within a tolerance."""
    rtol = 10 * get_eps(x) if rtol is None else rtol
    atol = 10 * get_eps(x) if atol is None else atol
    return tf.reduce_all(tf.abs(x - y) <= tf.abs(y) * rtol + atol)


def logm(inp):
    """Compute the matrix logarithm of positive-definite real matrices."""
    inp = tf.convert_to_tensor(inp)
    complex_dtype = tf.complex128 if inp.dtype == tf.float64 else tf.complex64
    log = tf.linalg.logm(tf.cast(inp, dtype=complex_dtype))
    return tf.cast(log, dtype=inp.dtype)


def transposem(inp):
    """Transpose multiple matrices."""
    perm = list(range(len(inp.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return tf.transpose(inp, perm)


def assign_to_manifold(var, manifold):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if not manifold.check_shape(var):
        raise ValueError("Invalid variable shape {}".format(var.shape))
    setattr(var, "manifold", manifold)


def get_manifold(var, default_manifold=Euclidean()):
    if not isinstance(var, tf.Variable):
        raise ValueError("var should be a TensorFlow variable")
    if hasattr(var, "manifold"):
        return var.manifold
    else:
        return default_manifold