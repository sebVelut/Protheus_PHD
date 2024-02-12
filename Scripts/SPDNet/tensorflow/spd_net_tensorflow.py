import numpy as np

import tensorflow as tf
from keras.layers import Input

from scikeras.wrappers import KerasClassifier

from .layers import BiMap, ReEig, LogEig

import os
import json

from sklearn import metrics
from typing import Any, Dict

class SPDNet_Tensorflow(KerasClassifier):

    def __init__(self, n_classes=2, bimap_dims=[60, 30, 15], eig_eps=1e-4, **kwargs):
        super(SPDNet_Tensorflow, self).__init__(**kwargs)
        self.bimap_dims = bimap_dims
        self.n_classes = n_classes
        self.eig_eps = eig_eps

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):

        model = tf.keras.models.Sequential()
        model.add(Input(shape=(self.X_shape_[1], self.X_shape_[2])))
        for output_dim in self.bimap_dims:
            model.add(BiMap(output_dim))
            model.add(ReEig(self.eig_eps))
        model.add(LogEig())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.n_classes, use_bias=False))

        model.compile(
            optimizer=self.optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()],)

        return model