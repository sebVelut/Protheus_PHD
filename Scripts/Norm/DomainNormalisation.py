import argparse
import numpy as np

import tensorflow as tf
from keras.src.engine import base_preprocessing_layer
from keras.layers import Input, Normalization

from scikeras.wrappers import KerasClassifier


import os
import json

from sklearn import metrics
from typing import Any, Dict

class DomainNormalisation(base_preprocessing_layer.PreprocessingLayer):
    def __init__(self):
        super.__init__(DomainNormalisation,self)

        self.domains = []
        self.normLayerList = {}
    
    def adapt(self,X,domains):
        for i,d in enumerate(domains.unique()):
            if d not in self.domains:
                self.add_domain(d)
            normi = Normalization(axis=-2)
            normi.adapt(X[domains==d])
            self.normLayerList[d] = normi
        
    def add_domain(self,d):
        self.domains.append(d)
        
    
    def call(self,X,domains,training=True):
        #doo the calling part
        for d in domains.unique():
            if d not in self.domains:
                raise argparse.ArgumentTypeError('A domain in domains was not train on at all')
            X[domains==d] = self.normLayerList[d](X[domains==d])
        return X