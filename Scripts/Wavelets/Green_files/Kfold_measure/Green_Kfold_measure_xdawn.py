import sys
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\Scripts")


import time
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from _utils import make_preds_accumul_aggresive
from pyriemann.estimation import Xdawn

from Wavelets.Green_files.green.wavelet_layers import RealCovariance
from Wavelets.Green_files.research_code.pl_utils import get_green, GreenClassifierLM

import torch
from SPDNet.SPD_torch.optimizers import riemannian_adam as torch_riemannian_adam
from torch.utils.data import DataLoader, TensorDataset
from utils import balance





class Green_Kfolder_Xdawn():
    def __init__(self,n_fold,epochs=20,batch_size=64,method=["SS","DA","DG"],n_ch=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_fold = n_fold
        self.n_ch = n_ch
        self.method = method


    def DG_Kfold(self,X_parent,Y_parent,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        
        accuracy_code_DG = np.zeros((self.n_fold,1))
        tps_train_code_DG = np.zeros((self.n_fold,1))
        tps_test_code_DG = np.zeros((self.n_fold,1))
        tps_pred_code_DG = np.zeros((self.n_fold,1))
        accuracy_DG = np.zeros((self.n_fold,1))

        for k in range(self.n_fold):
            print("Fold number ",k)
            print("TL to the participant : ", subtest)
            ind2take = [j for j in range(len(subjects)) if j!=subtest]
            X = X_parent.copy()
            Y = Y_parent.copy()
            domains = domains_parent.copy()


            X_train = np.concatenate(X[ind2take]).reshape(-1,X.shape[-2],X.shape[-1])
            Y_train = np.concatenate(Y[ind2take]).reshape(-1)
            domains_train = np.concatenate(domains[ind2take]).reshape(-1)
            X_test = X[subtest]
            Y_test = Y[subtest]
            labels_code_test = labels_codes[subtest]

            xdawn = Xdawn(nfilter=16,classes=[1],estimator='lwf')
            xdawn = xdawn.fit(X_train, Y_train)
            X_train = xdawn.transform(X_train)
            X_test = xdawn.transform(X_test)
            X_train = np.hstack([X_train,np.tile(xdawn.evokeds_[None,:,:],(X_train.shape[0],1,1))])
            X_test = np.hstack([X_test,np.tile(xdawn.evokeds_[None,:,:],(X_test.shape[0],1,1))])

            print("balancing the number of ones and zeros")
            X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)
            print(X_train.shape)
            print(Y_train.shape)
            print(X_test.shape)

            print("Creating the different pipelines")
            lr = 1e-3
            # optimizer = riemannian_adam.RiemannianAdam(learning_rate=lr)
            batchsize = 64 #128 # 64 for burst
            clf = get_green(
                            n_freqs=20,
                            kernel_width_s=window_size,
                            n_ch=self.n_ch,
                            sfreq=500,
                            orth_weights=False,
                            dropout=.7,
                            hidden_dim=[20,10],
                            logref='logeuclid',
                            pool_layer=RealCovariance(),
                            bi_out=[16,8],
                            dtype=torch.float32,
                            out_dim=2,
                            use_age=False,
                        )

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4
            
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

            # Convert data into PyTorch tensors
            X_train_tensor = torch.tensor(x_train, dtype=torch.float64)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(x_val, dtype=torch.float64)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
            y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

            # Create DataLoader for train, validation, and test sets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

            # Define loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch_riemannian_adam.RiemannianAdam(clf.parameters(), lr=lr)

            # Train the model
            num_epochs = 20

            for epoch in range(num_epochs):
                running_loss = 0.0
                train_y_pred= []
                y_train = []
                for inputs, labels in train_dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    # print(inputs.shape)
                    # print(labels.shape)
                    outputs = clf(inputs)
                    # print(outputs.get_device())
                    # print(labels.get_device())

                    labels = labels.to('cpu')
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    train_y_pred.append(predicted.to('cpu'))
                    y_train.append(labels.to('cpu'))

                    running_loss += loss.item()

                print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

            print("Training finished!")
            print("Training accuracy :", balanced_accuracy_score(np.concatenate(y_train),np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(train_y_pred)])))
            tps_train_code_DG[k] = time.time() - start

            # Validation
            clf.eval()
            val_correct = 0
            val_y_pred = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    val_correct += (predicted == labels).sum().item()
                    val_y_pred.append(predicted)


            val_accuracy = balanced_accuracy_score(y_val,np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(val_y_pred)]))#val_correct / len(x_val)
            print(f"Validation Accuracy: {val_accuracy}")

            # Testing
            start = time.time()
            test_correct = 0
            y_pred= []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    y_pred.append(predicted)
                    test_correct += (predicted == labels).sum().item()
                    
            test_accuracy = test_correct / len(X_test)
            
            # print("getting accuracy of participant ", i)
            test_y_pred = np.concatenate(y_pred)

            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in test_y_pred])
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

            # tn, fp, fn, tp = confusion_matrix(y_test_norm, y_pred_norm).ravel()
            accuracy_DG[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)
            print(f"Test Accuracy: {accuracy_DG[k]}")

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_DG[k] = time.time() - start
            accuracy_code_DG[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
            tps_pred_code_DG[k] = np.mean(mean_long_accumul)


        pd.DataFrame(accuracy_DG).to_csv("./results/score/{}_DG_score_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(accuracy_code_DG).to_csv("./results/score_code/{}_DG_score_code_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_train_code_DG).to_csv("./results/tps_train/{}_DG_tps_train_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_test_code_DG).to_csv("./results/tps_test/{}_DG_ps_test_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_pred_code_DG).to_csv("./results/tps_pred/{}_DG_ps_pred_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        return 0

    def Normal_Kfold(self,X_parent,Y_parent,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        accuracy_code_SS = np.zeros((self.n_fold,1))
        tps_train_code_SS = np.zeros((self.n_fold,1))
        tps_test_code_SS = np.zeros((self.n_fold,1))
        tps_pred_code_SS = np.zeros((self.n_fold,1))
        accuracy_SS = np.zeros((self.n_fold,1))

        n_cal = 7
        nb_samples_windows = int((2.2-window_size)*n_class*n_cal*60)

        for k in range(self.n_fold):
            print("fold number ",k)
            print("TL to the participant : ", subtest)
            X = X_parent.copy()
            Y = Y_parent.copy()
            domains = domains_parent.copy()

            X_train = X[subtest][:nb_samples_windows]
            Y_train = Y[subtest][:nb_samples_windows]
            domains_train = domains[subtest][:nb_samples_windows]

            X_test = X[subtest][nb_samples_windows:]
            Y_test = Y[subtest][nb_samples_windows:]
            labels_code_test = labels_codes[subtest][n_cal*n_class:]

            print(X_train.shape)
            print(Y_train.shape)
            print(X_test.shape)

            print("balancing the number of ones and zeros")
            X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)
            print(X_train.shape)
            print(Y_train.shape)
            print(X_test.shape)

            print("Creating the different pipelines")
            lr = 1e-3
            # optimizer = riemannian_adam.RiemannianAdam(learning_rate=lr)
            batchsize = 64 #128 # 64 for burst
            clf = get_green(
                n_freqs=20,
                kernel_width_s=window_size,
                n_ch=self.n_ch,
                sfreq=500,
                orth_weights=False,
                dropout=.7,
                hidden_dim=[20,10],
                logref='logeuclid',
                pool_layer=RealCovariance(),
                bi_out=[16,8],
                dtype=torch.float32,
                out_dim=2,
                use_age=False,
            )
            model_pl = GreenClassifierLM(model=clf,
                                        criterion=torch.nn.CrossEntropyLoss(),)

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4
            
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

            # Convert data into PyTorch tensors
            X_train_tensor = torch.tensor(x_train, dtype=torch.float64)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(x_val, dtype=torch.float64)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
            y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

            # Create DataLoader for train, validation, and test sets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

            # Define loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch_riemannian_adam.RiemannianAdam(clf.parameters(), lr=lr)

            # Train the model
            num_epochs = 20

            for epoch in range(num_epochs):
                running_loss = 0.0
                train_y_pred= []
                y_train = []
                for inputs, labels in train_dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    # print(inputs.shape)
                    # print(labels.shape)
                    outputs = clf(inputs)
                    # print(outputs.get_device())
                    # print(labels.get_device())

                    labels = labels.to('cpu')
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    train_y_pred.append(predicted.to('cpu'))
                    y_train.append(labels.to('cpu'))

                    running_loss += loss.item()

                print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

            print("Training finished!")
            print("Training accuracy :", balanced_accuracy_score(np.concatenate(y_train),np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(train_y_pred)])))
            tps_train_code_SS[k] = time.time() - start

            # Validation
            clf.eval()
            val_correct = 0
            val_y_pred = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    val_correct += (predicted == labels).sum().item()
                    val_y_pred.append(predicted)


            val_accuracy = balanced_accuracy_score(y_val,np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(val_y_pred)]))#val_correct / len(x_val)
            print(f"Validation Accuracy: {val_accuracy}")

            # Testing
            start = time.time()
            test_correct = 0
            y_pred= []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    y_pred.append(predicted)
                    test_correct += (predicted == labels).sum().item()
                    
            test_accuracy = test_correct / len(X_test)
            
            # print("getting accuracy of participant ", i)
            test_y_pred = np.concatenate(y_pred)
            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in test_y_pred])
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

            # tn, fp, fn, tp = confusion_matrix(y_test_norm, y_pred_norm).ravel()
            accuracy_SS[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)
            print(f"Test Accuracy: {accuracy_SS[k]}")

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_SS[k] = time.time() - start
            accuracy_code_SS[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
            tps_pred_code_SS[k] = np.mean(mean_long_accumul)


        pd.DataFrame(accuracy_SS).to_csv("./results/score/{}_SS_score_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(accuracy_code_SS).to_csv("./results/score_code/{}_SS_score_code_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_train_code_SS).to_csv("./results/tps_train/{}_SS_tps_train_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_test_code_SS).to_csv("./results/tps_test/{}_SS_ps_test_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_pred_code_SS).to_csv("./results/tps_pred/{}_SS_ps_pred_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        return 0

    def DA_Kfold(self,X_parent,Y_parent,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        accuracy_code_DA = np.zeros((self.n_fold,1))
        tps_train_code_DA = np.zeros((self.n_fold,1))
        tps_test_code_DA = np.zeros((self.n_fold,1))
        tps_pred_code_DA = np.zeros((self.n_fold,1))
        accuracy_DA = np.zeros((self.n_fold,1))

        n_cal = 4
        nb_samples_windows = int((2.2-window_size)*n_class*n_cal*60)

        for k in range(self.n_fold):
            print("fold number ",k)
            print("TL to the participant : ", subtest)
            ind2take = [j for j in range(len(subjects)) if j!=subtest]
            X = X_parent.copy()
            Y = Y_parent.copy()
            domains = domains_parent.copy()

            X_train = np.concatenate([np.concatenate(X[ind2take]),X[subtest][:nb_samples_windows]])
            Y_train = np.concatenate([np.concatenate(Y[ind2take]),Y[subtest][:nb_samples_windows]])
            domains_train = np.concatenate([np.concatenate(domains[ind2take]),domains[subtest][:nb_samples_windows]])

            X_test = X[subtest][nb_samples_windows:]
            Y_test = Y[subtest][nb_samples_windows:]
            labels_code_test = labels_codes[subtest][n_cal*n_class:]

            print("balancing the number of ones and zeros")
            X_train, Y_train, domains_train = balance(X_train,Y_train,domains_train)
            print(X_train.shape)
            print(Y_train.shape)
            print(X_test.shape)

            print("Creating the different pipelines")
            lr = 1e-3
            # optimizer = riemannian_adam.RiemannianAdam(learning_rate=lr)
            batchsize = 64 #128 # 64 for burst
            clf = get_green(
                            n_freqs=20,
                            kernel_width_s=window_size,
                            n_ch=self.n_ch,
                            sfreq=500,
                            orth_weights=False,
                            dropout=.7,
                            hidden_dim=[20,10],
                            logref='logeuclid',
                            pool_layer=RealCovariance(),
                            bi_out=[16,8],
                            dtype=torch.float32,
                            out_dim=2,
                            use_age=False,
                        )

            print("Fitting")
            start = time.time()
            weight_decay = 1e-4
            
            x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, shuffle=True)

            # Convert data into PyTorch tensors
            X_train_tensor = torch.tensor(x_train, dtype=torch.float64)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(x_val, dtype=torch.float64)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
            y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

            # Create DataLoader for train, validation, and test sets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

            # Define loss function and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch_riemannian_adam.RiemannianAdam(clf.parameters(), lr=lr)

            # Train the model
            num_epochs = 20

            for epoch in range(num_epochs):
                running_loss = 0.0
                train_y_pred= []
                y_train = []
                for inputs, labels in train_dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    # print(inputs.shape)
                    # print(labels.shape)
                    outputs = clf(inputs)
                    # print(outputs.get_device())
                    # print(labels.get_device())

                    labels = labels.to('cpu')
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    train_y_pred.append(predicted.to('cpu'))
                    y_train.append(labels.to('cpu'))

                    running_loss += loss.item()

                print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_dataloader)}")

            print("Training finished!")
            print("Training accuracy :", balanced_accuracy_score(np.concatenate(y_train),np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(train_y_pred)])))
            tps_train_code_DA[k] = time.time() - start

            # Validation
            clf.eval()
            val_correct = 0
            val_y_pred = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    val_correct += (predicted == labels).sum().item()
                    val_y_pred.append(predicted)


            val_accuracy = balanced_accuracy_score(y_val,np.array([1 if (y >= 0.5) else 0 for y in np.concatenate(val_y_pred)]))#val_correct / len(x_val)
            print(f"Validation Accuracy: {val_accuracy}")

            # Testing
            start = time.time()
            test_correct = 0
            y_pred= []
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    outputs = clf(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = predicted.to('cpu')
                    y_pred.append(predicted)
                    test_correct += (predicted == labels).sum().item()
                    
            test_accuracy = test_correct / len(X_test)
            
            # print("getting accuracy of participant ", i)
            test_y_pred = np.concatenate(y_pred)

            y_pred_norm = np.array([1 if (y >= 0.5) else 0 for y in test_y_pred])
            y_test_norm = np.array([0 if y == 0 else 1 for y in Y_test])

            # tn, fp, fn, tp = confusion_matrix(y_test_norm, y_pred_norm).ravel()
            accuracy_DA[k] = balanced_accuracy_score(y_test_norm,y_pred_norm)
            print(f"Test Accuracy: {accuracy_DA[k]}")

            labels_pred_accumul, _, mean_long_accumul = make_preds_accumul_aggresive(
                y_pred_norm, codes, min_len=30, sfreq=60, consecutive=50, window_size=window_size
            )
            tps_test_code_DA[k] = time.time() - start
            accuracy_code_DA[k] = np.round(balanced_accuracy_score(labels_code_test[labels_pred_accumul!=-1], labels_pred_accumul[labels_pred_accumul!=-1]), 2)
            tps_pred_code_DA[k] = np.mean(mean_long_accumul)


        pd.DataFrame(accuracy_DA).to_csv("./results/score/{}_DA_score_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(accuracy_code_DA).to_csv("./results/score_code/{}_DA_score_code_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_train_code_DA).to_csv("./results/tps_train/{}_DA_tps_train_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_test_code_DA).to_csv("./results/tps_test/{}_DA_ps_test_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        pd.DataFrame(tps_pred_code_DA).to_csv("./results/tps_pred/{}_DA_ps_pred_{}_{}_{}.csv".format("Green",window_size,prefix,subtest))
        return 0
    
    def perform_Kfold(self,X,Y,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class=4,window_size=0.25):
        if "SS" in self.method:
            print("Get score without transfer learning \n\n")
            self.Normal_Kfold(X,Y,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)
        if "DG" in self.method:
            print("Get score with DG transfer learning \n\n")
            self.DG_Kfold(X,Y,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)
        if "DA" in self.method:
            print("Get score with DA transfer learning \n\n")
            self.DA_Kfold(X,Y,domains_parent,labels_codes,codes,subjects,subtest,prefix,n_class,window_size)

        return 0

