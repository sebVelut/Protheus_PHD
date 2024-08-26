import abc
import tensorflow as tf
from .manifolds import StiefelEuclidean
from .manifolds import assign_to_manifold
from .manifolds import transposem
import numpy as np


def logm(inputs):
    s, u, v = tf.linalg.svd(inputs)
    log_s = tf.math.log(s)
    return u @ tf.linalg.diag(log_s) @ tf.transpose(v, perm=[0, 1, 2, 4, 3])

def expm(inputs):
    s, u, v = tf.linalg.svd(inputs)
    log_s = tf.math.exp(s)
    return u @ tf.linalg.diag(log_s) @ tf.transpose(v, perm=[0, 1, 3, 2])

@tf.keras.utils.register_keras_serializable(name="Centroid")
class CentroidLayer(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, in_ch=1, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(trainable=False, *args, **kwargs)

        self.in_ch = in_ch
        self.C = tf.Variable(np.zeros((2, self.in_ch, 32, 32), dtype=np.float32), trainable=False)
        self.C_init = False
    def call(self, X, idx, eta=0.01, training=None):

        if training:
            X_split = tf.gather(X, indices=idx, axis=0)
            if not self.C_init:
                self.C.assign(tf.reduce_mean(X_split, axis=0))
                self.C_init = True

            for j in range(1):

                s, u, v = tf.linalg.svd(self.C)
                C_m05 = u @ tf.linalg.diag(1 / (tf.sqrt(s) + 1.e-9)) @ tf.transpose(v, [0, 1, 3, 2])
                C_p05 = u @ tf.linalg.diag(tf.sqrt(s)) @ tf.transpose(v, [0, 1, 3, 2])

                M = tf.einsum('cnij,bcnjk->bcnik', C_m05, tf.einsum('bcnij,cnjk->bcnik', X_split, C_m05))
                L = logm(M)
                # print("In centroid layer, L is:",L)
                L_m = tf.reduce_mean(L, axis=0) * eta
                # print("In centroid layer, L_m is:",L_m)

                L_norm = tf.norm(L_m, axis=[1, 2])


                self.C.assign(tf.einsum('cnij,cnjk->cnik', C_p05, tf.einsum('cnij,cnjk->cnik', expm(L_m), C_p05)))
                eta *= 0.95

@tf.keras.utils.register_keras_serializable(name="Diag")
class DiagLayer(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)

    def call(self, X, C):

        s, u, v = tf.linalg.svd(C)
        diag = tf.linalg.diag_part(tf.einsum('cnij,bcnkj->bcnik', u, tf.einsum('bnij,cnjk->bcnik', X, u)))
        diag = tf.nn.relu(diag) + 1.e-5 
        return diag

@tf.keras.utils.register_keras_serializable(name="SpectralConv")
class SpectralConvLayer(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, n_centroids=2, in_ch=1, out_ch=4, n_diag=32, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.n_diag = n_diag
        self.n_centroids = n_centroids

    def build(self, input_shape):
        self.w = self.add_weight("w", shape=(self.n_diag, self.n_centroids, self.in_ch, self.out_ch),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 constraint=tf.keras.constraints.NonNeg())
        self.w = tf.abs(self.w)

    def call(self, D, C):
        s, u, v = tf.linalg.svd(C)
        c_diag = tf.linalg.diag(tf.einsum('bcnd,dcno->bcnod', D, self.w))
        spd = tf.einsum('cnij,bcnojk->bcnoik', u, tf.einsum('bcnoij, cnkj->bcnoik', c_diag, u))
        spd = tf.reduce_sum(spd, axis=[1, 2])

        return spd

@tf.keras.utils.register_keras_serializable(name="LogEig2")
class LogEig2(tf.keras.layers.Layer):
    """Eigen Log layer."""

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        log_s = tf.math.log(s)

        return u @ tf.linalg.diag(log_s) @ transposem(v)

'''
@tf.keras.utils.register_keras_serializable(name="BiMap")
class BiMap(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, output_dim, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            "w", shape=[int(input_shape[-1]), self.output_dim]
        )
        assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs):
        return transposem(self.w) @ inputs @ self.w

    def get_config(self):
        config = {"output_dim": self.output_dim}
        return dict(list(super().get_config().items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(name="BiMap2")
class BiMap2(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, output_dim, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            "w", shape=[int(input_shape[-1]), self.output_dim]
        )
        assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs):
        return (transposem(self.w) @ inputs @ self.w) * tf.eye(self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim}
        return dict(list(super().get_config().items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(name="BiMap3")
class BiMap3(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, output_dim, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.w = self.add_weight(
            "w", shape=[int(input_shape[-1])], constraint=tf.keras.constraints.NonNeg()
        )
        #assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs):

        s, u, v = tf.linalg.svd(inputs)

        sigma = s * self.w

        return transposem(v)[:, 0:self.output_dim, :] @ tf.linalg.diag(sigma) @ v[:, :, 0:self.output_dim]

    def get_config(self):
        config = {"output_dim": self.output_dim}
        return dict(list(super().get_config().items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(name="ReEig")
class ReEig(tf.keras.layers.Layer):
    """Eigen Rectifier layer."""

    def __init__(self, epsilon=1e-4, *args, **kwargs):
        """Instantiate the ReEig layer.

        Args:
          epsilon: a rectification threshold value
        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        sigma = tf.maximum(s, self.epsilon)
        return u @ tf.linalg.diag(sigma) @ transposem(v)

    def get_config(self):
        config = {"epsilon": self.epsilon}
        return dict(list(super().get_config().items()) + list(config.items()))

@tf.keras.utils.register_keras_serializable(name="LogEig")
class LogEig(tf.keras.layers.Layer):
    """Eigen Log layer."""

    def call(self, inputs):
        s, u, v = tf.linalg.svd(inputs)
        log_s = tf.math.log(s)

        return u @ tf.linalg.diag(log_s) @ transposem(v)

@tf.keras.utils.register_keras_serializable(name="LogEig2")
class LogEig2(tf.keras.layers.Layer):
    """Eigen Log layer."""

    def call(self, inputs):
        log_s = tf.math.log(inputs)

        return log_s
'''