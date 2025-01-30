import abc
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from .manifolds import StiefelEuclidean
from .manifolds import assign_to_manifold
from .manifolds import transposem
import numpy as np


def logm(inputs):
    eigval, eigvecs = tf.linalg.eigh(inputs)
    log_s = tf.math.log(tf.nn.relu(eigval) + 1.e-9)
    return eigvecs @ tf.linalg.diag(log_s) @ tf.transpose(eigvecs, perm=[0, 1, 3, 2])


def expm(inputs):
    eigval, eigvecs = tf.linalg.eigh(inputs)
    exp_s = tf.math.exp(eigval)
    return eigvecs @ tf.linalg.diag(exp_s) @ tf.transpose(eigvecs, perm=[0, 2, 1])


@tf.keras.utils.register_keras_serializable(name="CentroidLayer")
class CentroidLayer(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, n_centroids, in_ch, in_comp, out_comp, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(trainable=False, dtype=tf.float64, *args, **kwargs)

        self.n_centroids = n_centroids
        self.in_ch = in_ch
        self.in_comp = in_comp
        self.out_comp = out_comp
        self.C = tf.Variable(np.zeros((self.n_centroids, self.in_ch, self.in_comp, self.in_comp),
                             dtype=np.float64), trainable=False)
        self.eta = tf.Variable(0.1, dtype=tf.float64, trainable=False)

    def call(self, X, idx, training=None):

        if training:
            for i in range(self.n_centroids):

                tmp = tf.where(idx == i)[:, 0]
                X_split = tf.gather(X, indices=tf.where(idx == i)[:, 0], axis=0)

                is_empty = tf.equal(tf.size(tf.where(idx == i)[:, 0]), 0)
                if is_empty:
                    continue

                if tf.reduce_sum(self.C[i, :, :, :]) == 0:
                    self.C[i, :, :, :].assign(tf.reduce_mean(X_split, axis=0))

                eigvals, eigvecs = tf.linalg.eigh(self.C[i, :, :, :])
                eigvecs_t = tf.transpose(eigvecs, [0, 2, 1])
                s_sq = tf.sqrt(eigvals)
                C_m05 = eigvecs @ tf.linalg.diag(1 / (s_sq + 1.e-9)) @ eigvecs_t
                C_p05 = eigvecs @ tf.linalg.diag(s_sq) @ eigvecs_t

                M = tf.einsum('nij,bnjk->bnik', C_m05, tf.einsum('bnij,njk->bnik', X_split, C_m05))

                L_m = tf.reduce_mean(logm(M), axis=0) * self.eta
                C_update = C_p05 @ expm(L_m) @ C_p05
                self.C[i, :, :, :].assign(C_update)

            self.eta.assign(self.eta * 0.99)

        eigvals, eigvecs = tf.linalg.eigh(self.C)

        diag = tf.linalg.diag_part(tf.einsum('cnji,bcnjk->bcnik', eigvecs, tf.einsum('bnij,cnjk->bcnik', X, eigvecs)))

        return diag


@tf.keras.utils.register_keras_serializable(name="ConvLayerArithmetic")
class ConvLayerArithmetic(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, n_centroids=2, in_ch=1, out_ch=4, n_in=64, n_out=32, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_in = n_in
        self.n_out = n_out
        self.n_centroids = n_centroids

        self.w = tf.Variable(0.01 * np.abs(np.random.randn(self.n_in, self.n_centroids, self.in_ch, self.out_ch)),
                             trainable=True,
                             constraint=tf.keras.constraints.NonNeg(), dtype=tf.float32)

    def call(self, D, C):
        s, u, v = tf.linalg.svd(C)
        c_diag = tf.linalg.diag(tf.einsum('bcnd,dcno->bcnod', tf.cast(D, dtype=tf.float32), self.w))
        c_diag = tf.cast(c_diag, dtype=tf.float64)
        spd = tf.einsum('cnij,bcnojk->boik', u[:, :, 0:self.n_out, :], tf.einsum('bcnoij, cnkj->bcnoik', c_diag, u[:, :, 0:self.n_out, :]))

        return spd


@tf.keras.utils.register_keras_serializable(name="LogEigAJD")
class LogEigAJD(tf.keras.layers.Layer):
    """Eigen Log layer."""

    def call(self, inputs):
        return logm(inputs)
