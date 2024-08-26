import abc
import numpy as np
import tensorflow as tf
from keras.layers import BatchNormalization
from .manifolds import StiefelEuclidean
from .manifolds import assign_to_manifold
from .manifolds import transposem


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


@tf.keras.utils.register_keras_serializable(name="BiMap_2")
class BiMap_2(tf.keras.layers.Layer):
    """Bilinear Mapping layer."""

    def __init__(self, n_in, n_out, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.n_in = n_in
        self.n_out = n_out

    def build(self, input_shape):

        self.w = self.add_weight(
            "w", shape=[self.n_in],
            initializer=tf.random_uniform_initializer(minval=0.1, maxval=1),
            constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs):

        s, u, v = tf.linalg.svd(inputs)
        sigma = tf.einsum('bd,d->bd', s, self.w)
        return u[:, 0:self.n_out, :] @ tf.linalg.diag(sigma) @ transposem(v)[:, :, 0:self.n_out]

    def get_config(self):
        config = {"n_in": self.n_in, "n_out": self.n_out}
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
    

@tf.keras.utils.register_keras_serializable(name="DSNorm")
class DSNorm(tf.keras.layers.Layer):
    """Domain specific normalisation layer."""

    def __init__(self, *args, **kwargs):
        """Instantiate the BiMap layer.

        Args:
          output_dim: projection output dimension
        """
        super().__init__(*args, **kwargs)
        self.domains_called = []
        self.domains_norm = []

    def build(self, input_shape):
        self.w = self.add_weight(
            "w", shape=[int(input_shape[-1]), int(input_shape[-1])]
        )
        assign_to_manifold(self.w, StiefelEuclidean())

    def call(self, inputs, domains, training=None):
        if training : 
            for d in np.unique(domains):
                if d in self.domains_called:
                    X = self.domains_norm[self.domains_called.index(d)](inputs,training=training)
                else:
                    self.domains_called.append(d)
                    self.domains_norm.append(BatchNormalization())
                    # print("np.where",self.domains_called.index(d))
                    # print(self.domains_norm)
                    # print(self.domains_norm[self.domains_called.index(d)])
                    X = self.domains_norm[self.domains_called.index(d)](inputs,training=training)
        else:
            for d in np.unique(domains):
                if d in self.domains_called:
                    X = self.domains_norm[self.domains_called.index(d)](inputs,training=training)
                else:
                    print("A domain of the testing set is not pretrain")
        
        return X
