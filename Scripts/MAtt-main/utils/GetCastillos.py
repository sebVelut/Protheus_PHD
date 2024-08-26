import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import io
import os
import sys

sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD")
from Scripts.utils import balance, get_BVEP_data, prepare_data

   
# session 123 is training set, session4 is validation set, and session5 is testing set. 
def getAllDataloader(subject, ratio=8, data_path='', bs=64):
    dev = torch.device("cpu")
    subjects = [1,2,3,4,5,6,7,8,9,10,11,12]
    # subjects = [1,2,3,4]
    n_channels = 32
    on_frame = True

    raw_data,labels,codes,labels_codes = get_BVEP_data(subjects,on_frame)
    recenter = False
    toSPD = False

    tempdata,templabel,tempdomains = prepare_data(subjects,raw_data,labels,on_frame,toSPD,recenter=recenter,codes=codes)
    # print("shape vlid data",tempdata[2].shape)
    # print("shape vlid label",templabel[2].shape)
    # print("label",np.concatenate(templabel[:]).shape)
    temp_data_train,temp_label_train,_ = balance(np.concatenate(tempdata[:10]),np.concatenate(templabel[:10]),
                                                 np.concatenate(tempdomains[:10]))
    temp_data_valid,temp_label_valid,_ = balance(tempdata[10],templabel[10],tempdomains[10])

    x_train=torch.tensor(np.expand_dims(temp_data_train,1))
    y_train=torch.tensor(temp_label_train)
    
    x_valid=torch.tensor(np.expand_dims(temp_data_valid,1))
    y_valid=torch.tensor(temp_label_valid)
    
    x_test =torch.tensor(np.expand_dims(tempdata[11],1))
    y_test =torch.tensor(templabel[11])

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid.to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test.to(dev)
    y_test = y_test.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)
    

    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset = Data.TensorDataset(x_test, y_test)
    
    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size =bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )
    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader, testloader
    