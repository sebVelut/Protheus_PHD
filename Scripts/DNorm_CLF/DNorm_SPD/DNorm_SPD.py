import numpy as np

import tensorflow as tf
from keras.layers import Input
from sklearn.model_selection import train_test_split
from keras import Model

from scikeras.wrappers import KerasClassifier

from .layers import BiMap, DSNorm, ReEig, LogEig
from .optimizer.riemannian_adam import RiemannianAdam

import os
import json

from sklearn import metrics
from typing import Any, Dict


class SPDNet_Tensorflow(Model):

    def __init__(self, n_classes=2, bimap_dims=[60, 30, 15], eig_eps=1e-4, lr=0.001, **kwargs):
        super(SPDNet_Tensorflow, self).__init__(**kwargs)
        self.bimap_dims = bimap_dims
        self.n_classes = n_classes
        self.eig_eps = eig_eps
        self.lr = lr

        # self.input = Input(shape=(self.X_shape_[1], self.X_shape_[2]))
        self.bimap = []
        self.reeig = []
        for output_dim in self.bimap_dims:
            self.bimap.append(BiMap(output_dim))
            self.reeig.append(ReEig(self.eig_eps))
        
        self.logeig = LogEig()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n_classes, use_bias=False)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = RiemannianAdam(self.lr)


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
    
    def __call__(self, inputs):

        # i = self.input(inputs)
        s = inputs
        for k in range(len(self.bimap_dims)):
            s = self.bimap[k](s)
            s = self.reeig[k](s)
        l = self.logeig(s)

        f = self.flatten(l)
        c = self.dense(f)

        return c

class BNSPD_Net(Model):
    def __init__(self, batch_size=32, n_epochs=20, valid_split=0.2, 
                 n_classes=2, bimap_dims=[32,16,8], lr=0.001, **kwargs):
        super(BNSPD_Net, self).__init__(**kwargs)

        self.valid_split = valid_split
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.eta = 1.0
        self.bimap_dims = bimap_dims
        self.c1_estim = True
        self.n_classes = n_classes

        self.norm = DSNorm()
        self.net = SPDNet_Tensorflow(bimap_dims=self.bimap_dims)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = RiemannianAdam(self.lr)

    def __call__(self, X, domains, y=None, training=None):

        # X = tf.expand_dims(X, axis=1)
        N_1 = self.norm(X,domains,training=training)
        c = self.net(N_1)

        return c

    # @tf.function
    def train_step(self, X_, domains_, y_):
        with tf.GradientTape() as tf_tape:
            y_hat = self(X_, domains_, training=True)
            loss = self.loss(y_, y_hat)
        grads = tf_tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    # @tf.function
    def valid_step(self, data_batch, domains_batch, data_labs=None):
        estims = self(data_batch, domains_batch, training=False)

        return estims

    def fit_step(self, X, y, domains, output_path):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        

        # X /= 1000
        # X_test /= 1000
        split_p = int(np.floor(X.shape[0] * (1.0 - self.valid_split)))

        X_train, X_valid, domains_train, domains_valid, y_train, y_valid = train_test_split(X, domains, y, test_size=0.2, random_state=42, shuffle=True)

        train_data_tf = tf.data.Dataset.from_tensor_slices((X_train, domains_train, y_train)).shuffle(X_train.shape[0]).batch(self.batch_size)
        valid_data_tf = tf.data.Dataset.from_tensor_slices((X_valid, domains_valid, y_valid)).batch(self.batch_size)

        valid_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
        valid_acc = np.zeros(self.n_epochs, dtype=np.float32)
        valid_loss = np.zeros(self.n_epochs, dtype=np.float32)

        for e in range(self.n_epochs):
            
            print("calcul of loss")
            print(y_train[0:15])
            running_loss = 0.0
            for t_idx, (t_batch, t_dom, t_labs) in enumerate(train_data_tf):
                running_loss += self.train_step(t_batch,
                                                t_dom,
                                                t_labs)
            print(f"Epoch {e+1}, Loss: {running_loss / len(train_data_tf)}")


            y_estims = np.zeros(y_valid.shape[0], dtype=np.float32)
            for v_idx, (v_batch, v_doms, v_labs) in enumerate(valid_data_tf):
                v_estims = self.valid_step(v_batch, v_doms)
                v_estims = np.exp(v_estims) / np.expand_dims(np.sum(np.exp(v_estims), axis=1), axis=1)
                # print("v_estim :",v_estims)
                vs, ve = v_idx * self.batch_size, (v_idx + 1) * self.batch_size
                y_estims[vs:ve] = v_estims[:, 1]
            print("la prediction est:")
            print(y_estims[0:64])
            print(y_valid[0:64])
            valid_roc_auc[e] = metrics.roc_auc_score(y_valid, y_estims)
            valid_acc[e] = np.mean(y_valid== np.asarray(y_estims>0.5, dtype=np.int32))
            valid_loss[e] = -np.mean(y_valid * np.log(y_estims) + (1 - y_valid) * np.log(1 - y_estims))
            print("Valid acc, roc auc, loss:", valid_acc[e], valid_roc_auc[e], valid_loss[e])


            if e >= 100 and e % 10 == 0:
                lr_c = self.optimizer.lr.read_value()
                self.optimizer.lr.assign(lr_c * 0.95)
                print(e, lr_c, lr_c * 0.95)
        # np.save(os.path.join(output_path, 'valid_acc.npy'), valid_acc)
        # np.save(os.path.join(output_path, 'valid_roc_auc.npy'), valid_roc_auc)
        # np.save(os.path.join(output_path, 'valid_loss.npy'), valid_loss)
        # np.save(os.path.join(output_path, 'test_auc_roc.npy'), test_roc_auc[np.argmin(valid_loss)])

    def predict_step(self, X_test, domains_test, y_test):
        test_data_tf = tf.data.Dataset.from_tensor_slices((X_test,domains_test)).batch(self.batch_size)

        test_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
        test_acc = np.zeros(self.n_epochs, dtype=np.float32)
        
        y_estims = np.zeros(y_test.shape[0], dtype=np.float32)
        for test_idx, (test_batch, domains_batch) in enumerate(test_data_tf):
            test_estims = self.valid_step(test_batch,domains_batch)
            test_estims = np.exp(test_estims) / np.expand_dims(np.sum(np.exp(test_estims), axis=1), axis=1)
            ts, te = test_idx * self.batch_size, (test_idx + 1) * self.batch_size
            y_estims[ts:te] = test_estims[:, 1]
        test_roc_auc = metrics.roc_auc_score(y_test, y_estims)
        test_acc = np.mean(y_test == np.asarray(y_estims>0.5, dtype=np.int32))
        print("accuracy and rau score",(test_acc,test_roc_auc))

        return y_estims






# class BNSPD_Module(nn.Module):
#     def __init__(self, n_classes=2, bimap_dims=[64, 32], domain_target=None):
#         super(BNSPD_Module, self).__init__()
#         a=0

#         self.bn = nn.BatchNorm2d(1)
#         self.optimizer = optim.Adam(self.bn.parameters(), lr=0.001)
#         self.criterion = nn.CrossEntropyLoss()

#     def __call__(self, X, y=None, c=None, training=None):
#         l=0
#         return l

#     # @tf.function
#     def train_step(self, X_, y_, c_):
#         with tf.GradientTape() as tf_tape:
#             y_hat = self(X_, y_, c_, training=True)
#             loss = self.loss(y_, y_hat)
#         grads = tf_tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#         return loss

#     # @tf.function
#     def valid_step(self, data_batch, data_labs=None):
#         estims = self(data_batch, training=False)

#         return estims

#     def fit_and_predict(self, X, y, X_test, y_test, output_path):

#         if not os.path.exists(output_path):
#             os.makedirs(output_path)
        

#         X /= 1000
#         X_test /= 1000
#         split_p = int(np.floor(X.shape[0] * (1.0 - self.valid_split)))

#         # X_train = X[0:split_p, :, :].astype(np.float32)

#         X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
#         c_train = y_train.copy()
#         # y_train = y[0:split_p]
#         # c_train = y[0:split_p]

#         # X_valid = X[split_p:, :, :].astype(np.float32)
#         # y_valid = y[split_p:]

#         train_data_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(self.batch_size)
#         valid_data_tf = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)
#         test_data_tf = tf.data.Dataset.from_tensor_slices(X_test).batch(self.batch_size)
#         C = np.zeros((self.n_classes * self.n_centroids,
#                       X_train.shape[1], X_train.shape[2]), dtype=X_train.dtype)
#         for i in range(self.n_classes):
#             C[i, :, :] = np.mean(X_train[y_train == i, :, :], axis=0)

#         self.C = tf.Variable(tf.convert_to_tensor(C, dtype=tf.float32), trainable=False)

#         valid_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
#         valid_acc = np.zeros(self.n_epochs, dtype=np.float32)
#         valid_loss = np.zeros(self.n_epochs, dtype=np.float32)

#         test_roc_auc = np.zeros(self.n_epochs, dtype=np.float32)
#         test_acc = np.zeros(self.n_epochs, dtype=np.float32)

#         for e in range(self.n_epochs):

#             t_centroid_idx = np.zeros((X_train.shape[0], self.n_classes * self.n_centroids), dtype=np.int32)
#             count = np.zeros(self.n_classes * self.n_centroids, dtype=np.int32)
#             for tc_idx, tc in enumerate(c_train):
#                 t_centroid_idx[count[tc], tc] = tc_idx
#                 count[tc] += 1
#             t_centroid_idx = t_centroid_idx[0:np.min(count), :]
            
#             print("calcul of loss")
#             print(y_train[0:15])

#             loss = self.train_step(tf.convert_to_tensor(X_train),
#                                    tf.convert_to_tensor(y_train),
#                                    tf.convert_to_tensor(t_centroid_idx))
#             print("Epoch:", e)
#             print("Train loss:", loss)
#             y_estims = np.zeros(y_valid.shape[0], dtype=np.float32)
#             for v_idx, (v_batch, v_labs) in enumerate(valid_data_tf):
#                 v_estims = self.valid_step(v_batch)
#                 v_estims = np.exp(v_estims) / np.expand_dims(np.sum(np.exp(v_estims), axis=1), axis=1)
#                 # print("v_estim :",v_estims)
#                 vs, ve = v_idx * self.batch_size, (v_idx + 1) * self.batch_size
#                 y_estims[vs:ve] = v_estims[:, 1]
#             print("la prediction est:")
#             print(y_estims[0:64])
#             print(y_valid[0:64])
#             valid_roc_auc[e] = roc_auc_score(y_valid, y_estims)
#             valid_acc[e] = np.mean(y_valid== np.asarray(y_estims>0.5, dtype=np.int32))
#             valid_loss[e] = -np.mean(y_valid * np.log(y_estims) + (1 - y_valid) * np.log(1 - y_estims))
#             print("Valid acc, roc auc, loss:", valid_acc[e], valid_roc_auc[e], valid_loss[e])

#             y_estims = np.zeros(y_test.shape[0], dtype=np.float32)
#             for test_idx, test_batch in enumerate(test_data_tf):
#                 test_estims = self.valid_step(test_batch)
#                 test_estims = np.exp(test_estims) / np.expand_dims(np.sum(np.exp(test_estims), axis=1), axis=1)
#                 ts, te = test_idx * self.batch_size, (test_idx + 1) * self.batch_size
#                 y_estims[ts:te] = test_estims[:, 1]
#             test_roc_auc[e] = roc_auc_score(y_test, y_estims)
#             test_acc[e] = np.mean(y_test == np.asarray(y_estims>0.5, dtype=np.int32))

#             if e >= 100 and e % 10 == 0:
#                 lr_c = self.optimizer.lr.read_value()
#                 self.optimizer.lr.assign(lr_c * 0.95)
#                 print(e, lr_c, lr_c * 0.95)
#         np.save(os.path.join(output_path, 'valid_acc.npy'), valid_acc)
#         np.save(os.path.join(output_path, 'valid_roc_auc.npy'), valid_roc_auc)
#         np.save(os.path.join(output_path, 'valid_loss.npy'), valid_loss)
#         np.save(os.path.join(output_path, 'test_auc_roc.npy'), test_roc_auc[np.argmin(valid_loss)])

#     def predict(self, X):
#         pass

    
#     def fit(self,X,domains,Y):
#         """
#         Fit the normalization layer and transform the data in X, fit only a mini batch for domain_target
#         Then fit the clf
#         """
#         du = domains.unique()
        
#         for d in du:
#             # define the trainloader for each domain
            
#             train_dataset = 0
#             train_dataloader = 0


#             # Fit the normalization et transforme les inputs en premier
#             if self.training:
#                 for epoch in range(2):
#                     running_loss = 0.0
#                     for inputs, labels in train_dataloader:
#                         # Zero the parameter gradients
#                         self.optimizer.zero_grad()

#                         # Forward pass
#                         outputs = self.bn(inputs)
#                         loss = self.criterion(outputs, labels)

#                         # Backward pass and optimize
#                         loss.backward()
#                         self.optimizer.step()
                

#             # Save the batchnorm layer for test domain

#         # Fit the SPDNet
    
#     def predict(X):
#         """
#         Normalize X and predict the labels then return it
#         """
#         # transform the target features

#         # predict the new labels