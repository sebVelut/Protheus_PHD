import os
import numpy as np

import tensorflow as tf

from tensorflow.keras import Model
from .layers_AJD import CentroidLayer, ConvLayerArithmetic, LogEigAJD

from sklearn.metrics import roc_auc_score
import pyriemann


class SPDNet_Tensorflow_AJD(Model):
    def __init__(self, batch_size=32, n_epochs=300, valid_split=0.2,
                 n_classes=2, n_centroids=2, lr=0.001, **kwargs):
        super(SPDNet_Tensorflow_AJD, self).__init__(**kwargs)

        self.valid_split = valid_split
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_centroids = n_centroids
        self.lr = lr

        self.centroid1 = CentroidLayer(self.n_centroids, 1, 64, 64)
        self.conv1 = ConvLayerArithmetic(self.n_centroids, in_ch=1, out_ch=2, n_in=64, n_out=32)
        self.centroid2 = CentroidLayer(self.n_centroids, 2, 32, 32)
        self.conv2 = ConvLayerArithmetic(self.n_centroids, in_ch=2, out_ch=4, n_in=32, n_out=16)
        self.centroid3 = CentroidLayer(self.n_centroids, 4, 16, 16)
        self.conv3 = ConvLayerArithmetic(self.n_centroids, in_ch=4, out_ch=8, n_in=16, n_out=8)

        self.logeig = LogEigAJD()

        self.flatten = tf.keras.layers.Flatten()
        self.fcn1 = tf.keras.layers.Dense(self.n_classes)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def __call__(self, X, y=None, c=None, training=None):

        X = tf.expand_dims(X, axis=1)

        D_1 = self.centroid1(X, c, training=training)
        S_1 = self.conv1(D_1, self.centroid1.C)
        D_2 = self.centroid2(S_1, c, training=training)
        S_2 = self.conv2(D_2, self.centroid2.C)
        D_3 = self.centroid3(S_2, c, training=training)
        S_3 = self.conv3(D_3, self.centroid3.C)

        feats = self.flatten(self.logeig(S_3))
        c = self.fcn1(feats)

        return c

    @tf.function
    def train_step(self, X_, y_, c_):
        with tf.GradientTape() as tf_tape:
            y_hat = self(X_, y_, c_, training=True)
            loss = self.loss(y_, y_hat)

        grads = tf_tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    @tf.function
    def valid_step(self, data_batch, data_labs=None):
        estims = self(data_batch, training=False)

        return estims

    def fit_and_predict(self, X, y, X_test, y_test, output_path):

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        X /= 100
        X_test /= 100
        Cm_pyriemann_0 = pyriemann.utils.mean.mean_covariance(X[np.where(y == 0)[0], :, :])
        Cm_pyriemann_1 = pyriemann.utils.mean.mean_covariance(X[np.where(y == 1)[0], :, :])

        split_p = int(np.floor(X.shape[0] * (1.0 - self.valid_split)))

        X_train = X[0:split_p, :, :].astype(np.float64)
        y_train = np.copy(y[0:split_p])
        c_train = np.copy(y[0:split_p])

        nc_per_class = self.n_centroids // self.n_classes
        for i in range(self.n_classes):
            class_idx = np.asarray(np.where(y_train == i)[0])
            ns_per_centre = int(len(class_idx) / nc_per_class)
            for j in range(nc_per_class - 1):
                c_train[class_idx[j * ns_per_centre:(j + 1) * ns_per_centre]] = i * nc_per_class + j
            c_train[class_idx[(nc_per_class - 1) * ns_per_centre:]] = (i + 1) * nc_per_class - 1

        X_valid = X[split_p:, :, :].astype(np.float64)
        y_valid = y[split_p:]

        train_data_tf = tf.data.Dataset.from_tensor_slices((X_train, y_train, c_train)).shuffle(X_train.shape[0]).batch(self.batch_size)
        valid_data_tf = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(self.batch_size)
        test_data_tf = tf.data.Dataset.from_tensor_slices(X_test).batch(self.batch_size)

        valid_roc_auc = np.zeros(self.n_epochs, dtype=np.float64)
        valid_acc = np.zeros(self.n_epochs, dtype=np.float64)
        valid_loss = np.zeros(self.n_epochs, dtype=np.float64)

        test_roc_auc = np.zeros(self.n_epochs, dtype=np.float64)
        test_acc = np.zeros(self.n_epochs, dtype=np.float64)

        init_c = False
        for e in range(self.n_epochs):
            for t_idx, (t_batch, t_labs, t_centre) in enumerate(train_data_tf):
                if not init_c:
                    n_cntrs = [np.sum(t_centre == c) for c in range(self.n_centroids)]
                    print(n_cntrs)
                    if not np.min(n_cntrs):
                        continue
                    else:
                        init_c = True
                loss = self.train_step(t_batch, t_labs, t_centre)
            print("Epoch:", e)
            print("Train loss:", loss)

            y_estims = np.zeros(y_valid.shape[0], dtype=np.float64)
            for v_idx, (v_batch, v_labs) in enumerate(valid_data_tf):
                v_estims = self.valid_step(v_batch)
                v_estims = np.exp(v_estims) / (np.expand_dims(np.sum(np.exp(v_estims), axis=1), axis=1) + 1.e-9)
                vs, ve = v_idx * self.batch_size, (v_idx + 1) * self.batch_size
                y_estims[vs:ve] = v_estims[:, 1]
            try:
                valid_roc_auc[e] = roc_auc_score(y_valid, y_estims)
            except:
                pass
            valid_acc[e] = np.mean(y_valid == np.asarray(y_estims > 0.5, dtype=np.int32))
            valid_loss[e] = -np.mean(y_valid * np.log(y_estims + 1.e-9) + (1 - y_valid) * np.log(1 - y_estims + 1.e-9))
            print("Valid acc, roc auc, loss:", valid_acc[e], valid_roc_auc[e], valid_loss[e])

            y_estims = np.zeros(y_test.shape[0], dtype=np.float64)
            for test_idx, test_batch in enumerate(test_data_tf):
                test_estims = self.valid_step(test_batch)
                test_estims = np.exp(test_estims) / (np.expand_dims(np.sum(np.exp(test_estims), axis=1), axis=1) + 1.e-9)
                ts, te = test_idx * self.batch_size, (test_idx + 1) * self.batch_size
                y_estims[ts:te] = test_estims[:, 1]
            try:
                test_roc_auc[e] = roc_auc_score(y_test, y_estims)
            except:
                pass
            test_acc[e] = np.mean(y_test == np.asarray(y_estims > 0.5, dtype=np.int32))
            if e >= 5 and np.mean(valid_loss[e - 4:e - 1]) < np.mean(valid_loss[e - 2:e + 1]) :
                lr_c = self.optimizer.lr.read_value()
                self.optimizer.lr.assign(lr_c * 0.99)
                print(e, lr_c, lr_c * 0.99)

        np.save(os.path.join(output_path, 'valid_acc.npy'), valid_acc)
        np.save(os.path.join(output_path, 'valid_roc_auc.npy'), valid_roc_auc)
        np.save(os.path.join(output_path, 'valid_loss.npy'), valid_loss)
        np.save(os.path.join(output_path, 'test_auc_roc.npy'), test_roc_auc[np.argmin(valid_loss)])

    def predict(self, X):
        pass
