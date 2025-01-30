import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import Input, Concatenate

from tensorflow.keras import Model
from .layers_test import CentroidLayer, DiagLayer, SpectralConvLayer, LogEig2


import os
import json

from sklearn import metrics
from typing import Any, Dict
from .manifolds import transposem
import scipy
from sklearn.metrics import roc_auc_score

class SPDNet_AJD(Model):
    def __init__(self, batch_size=32, n_epochs=300, valid_split=0.2, 
                 n_classes=2, n_centroids=1, lr=0.001, **kwargs):
        super(SPDNet_AJD, self).__init__(**kwargs)

        self.valid_split = valid_split
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eta = 1.0

        self.c1_estim = True

        self.n_classes = n_classes
        self.n_centroids = n_centroids

        self.c_1 = CentroidLayer(in_ch=1)
        self.d_1 = DiagLayer()
        self.s_1 = SpectralConvLayer(n_centroids=2, in_ch=1, out_ch=2, n_diag=32)

        self.c_2 = CentroidLayer(in_ch=2)
        self.d_2 = DiagLayer()
        self.s_2 = SpectralConvLayer(n_centroids=2, in_ch=2, out_ch=4, n_diag=32)

        self.c_3 = CentroidLayer(in_ch=4)
        self.d_3 = DiagLayer()
        self.s_3 = SpectralConvLayer(n_centroids=2, in_ch=4, out_ch=8, n_diag=32)

        self.log_eig = LogEig2()
        self.flatten = tf.keras.layers.Flatten()
        self.fcn1 = tf.keras.layers.Dense(self.n_classes, kernel_constraint=tf.keras.constraints.MaxNorm(0.1, axis=0))

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def __call__(self, X, y=None, c=None, training=None):

        X = tf.expand_dims(X, axis=1)
        self.c_1(X, c, training=training)
        D_1 = self.d_1(X, self.c_1.C)
        S_1 = self.s_1(D_1, self.c_1.C)

        self.c_2(S_1, c, training=training)
        D_2 = self.d_2(S_1, self.c_2.C)
        S_2 = self.s_2(D_2, self.c_2.C)

        self.c_3(S_2, c, training=training)
        D_3 = self.d_3(S_2, self.c_3.C)

        c = self.fcn1(self.flatten(D_3))

        return c

    # @tf.function
    def train_step(self, X_, y_, c_):
        with tf.GradientTape() as tf_tape:
            y_hat = self(X_, y_, c_, training=True)
            loss = self.loss(y_, y_hat)
        grads = tf_tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    # @tf.function
    def valid_step(self, data_batch, data_labs=None):
        estims = self(data_batch, training=False)

        return estims

    def fit_and_predict(self, X, y, X_test, y_test, output_path):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        

        X /= 1000
        X_test /= 1000
        split_p = int(np.floor(X.shape[0] * (1.0 - self.valid_split)))

        # X_train = X[0:split_p, :, :].astype(np.float32)

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        c_train = y_train.copy()
        # y_train = y[0:split_p]
        # c_train = y[0:split_p]

        # X_valid = X[split_p:, :, :].astype(np.float32)
        # y_valid = y[split_p:]

        train_data_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(self.batch_size)
        valid_data_tf = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)
        test_data_tf = tf.data.Dataset.from_tensor_slices(X_test).batch(self.batch_size)
        C = np.zeros((self.n_classes * self.n_centroids,
                      X_train.shape[1], X_train.shape[2]), dtype=X_train.dtype)
        for i in range(self.n_classes):
            C[i, :, :] = np.mean(X_train[y_train == i, :, :], axis=0)

        self.C = tf.Variable(tf.convert_to_tensor(C, dtype=tf.float32), trainable=False)

        valid_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
        valid_acc = np.zeros(self.n_epochs, dtype=np.float32)
        valid_loss = np.zeros(self.n_epochs, dtype=np.float32)

        test_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
        test_acc = np.zeros(self.n_epochs, dtype=np.float32)

        for e in range(self.n_epochs):

            t_centroid_idx = np.zeros((X_train.shape[0], self.n_classes * self.n_centroids), dtype=np.int32)
            count = np.zeros(self.n_classes * self.n_centroids, dtype=np.int32)
            for tc_idx, tc in enumerate(c_train):
                t_centroid_idx[count[tc], tc] = tc_idx
                count[tc] += 1
            t_centroid_idx = t_centroid_idx[0:np.min(count), :]
            
            print("calcul of loss")
            print(y_train[0:15])

            loss = self.train_step(tf.convert_to_tensor(X_train),
                                   tf.convert_to_tensor(y_train),
                                   tf.convert_to_tensor(t_centroid_idx))
            print("Epoch:", e)
            print("Train loss:", loss)
            y_estims = np.zeros(y_valid.shape[0], dtype=np.float32)
            for v_idx, (v_batch, v_labs) in enumerate(valid_data_tf):
                v_estims = self.valid_step(v_batch)
                v_estims = np.exp(v_estims) / np.expand_dims(np.sum(np.exp(v_estims), axis=1), axis=1)
                # print("v_estim :",v_estims)
                vs, ve = v_idx * self.batch_size, (v_idx + 1) * self.batch_size
                y_estims[vs:ve] = v_estims[:, 1]
            print("la prediction est:")
            print(y_estims[0:64])
            print(y_valid[0:64])
            valid_roc_auc[e] = roc_auc_score(y_valid, y_estims)
            valid_acc[e] = np.mean(y_valid== np.asarray(y_estims>0.5, dtype=np.int32))
            valid_loss[e] = -np.mean(y_valid * np.log(y_estims) + (1 - y_valid) * np.log(1 - y_estims))
            print("Valid acc, roc auc, loss:", valid_acc[e], valid_roc_auc[e], valid_loss[e])

            y_estims = np.zeros(y_test.shape[0], dtype=np.float32)
            for test_idx, test_batch in enumerate(test_data_tf):
                test_estims = self.valid_step(test_batch)
                test_estims = np.exp(test_estims) / np.expand_dims(np.sum(np.exp(test_estims), axis=1), axis=1)
                ts, te = test_idx * self.batch_size, (test_idx + 1) * self.batch_size
                y_estims[ts:te] = test_estims[:, 1]
            test_roc_auc[e] = roc_auc_score(y_test, y_estims)
            test_acc[e] = np.mean(y_test == np.asarray(y_estims>0.5, dtype=np.int32))

            if e >= 100 and e % 10 == 0:
                lr_c = self.optimizer.lr.read_value()
                self.optimizer.lr.assign(lr_c * 0.95)
                print(e, lr_c, lr_c * 0.95)
        np.save(os.path.join(output_path, 'valid_acc.npy'), valid_acc)
        np.save(os.path.join(output_path, 'valid_roc_auc.npy'), valid_roc_auc)
        np.save(os.path.join(output_path, 'valid_loss.npy'), valid_loss)
        np.save(os.path.join(output_path, 'test_auc_roc.npy'), test_roc_auc[np.argmin(valid_loss)])

    def predict(self, X):
        pass

'''
def estimate_centroids(self, X_train, y_train, eps=1.e-9, eta=1., tau=1.e9):

    C = np.zeros((self.n_classes * self.n_centroids,
                  X_train.shape[1], X_train.shape[2]), dtype=X_train.dtype)

    for i in range(self.n_classes):
        S_i = X_train[y_train == i, :, :]

        C_i = np.mean(S_i, axis=0)
        for r in range(10):
            C_i_m05 = scipy.linalg.fractional_matrix_power(C_i, -0.5)
            C_i_p05 = scipy.linalg.fractional_matrix_power(C_i, 0.5)
            M_i = np.einsum('ij,bjk->bik', C_i_m05, np.einsum('bij,jk->bik', S_i, C_i_m05))
            L_i = np.zeros((X_train.shape[1], X_train.shape[2]), dtype=np.float32)
            for j in range(M_i.shape[0]):
                L_i += scipy.linalg.logm(M_i[j, :, :])
            L_i *= eta / M_i.shape[0]
            L_i_n = scipy.linalg.norm(L_i)
            print(L_i_n)
            C_i = np.einsum('ij,jk->ik', C_i_p05, np.einsum('ij,jk->ik', scipy.linalg.expm(L_i), C_i_p05))

            if L_i_n < tau:
                C_i = np.einsum('ij,jk->ik', C_i_p05, np.einsum('ij,jk->ik', scipy.linalg.expm(L_i), C_i_p05))
                eta *= 1.0
                tau = L_i_n
            else:
                eta *= 0.5

        C[i, :, :] = C_i
    return C
'''