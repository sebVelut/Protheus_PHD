import numpy as np
import tensorflow as tf
from tensorflow import keras
from EEG2CodeKeras import basearchi
from Kfold_measure import Kfolder
from SPDNet.tensorflow.spd_net_tensorflow import SPDNet_Tensorflow
from SPDNet.torch.spd_net_bn_torch import SPDNetBN_Module
from utils import get_BVEP_data, prepare_data
import sys
import argparse
import os
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40


def main(subjects,subtest,on_frame,recenter):
    if not os.path.exists("./results/score/"):
        os.makedirs("./results/score/")
    if not os.path.exists("./results/score_code/"):
        os.makedirs("./results/score_code/")
    if not os.path.exists("./results/tps_train/"):
        os.makedirs("./results/tps_train/")
    if not os.path.exists("./results/tps_test/"):
        os.makedirs("./results/tps_test/")

    subjects = eval(subjects)
    subtest = subtest-1
    print(on_frame)
    print(recenter)
    # tf.debugging.set_log_device_placement(True)
    # print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
    # on_frame = eval(on_frame)
    # recenter = eval(recenter)
    prefix = ""
    if recenter:
        prefix = "recentered"

    # keep = ["O1", "O2", "Oz", "P7", "P3", "P4", "P8", "Pz","stim_trial","stim_epoch"]
    keep = None

    moabb_ds = CasitllosBurstVEP100()

    raw_data,labels,codes,labels_codes = get_BVEP_data(subjects,on_frame,to_keep=keep,moabb_ds=moabb_ds)
    X_parent_cnn,Y_parent_cnn,domains_parent_cnn = prepare_data(subjects,raw_data,labels,on_frame,False,recenter,codes=codes, normalise=True)
    

    # CNN Kfolder
    print("Perform CNN\n")
    cnn_kf = Kfolder("CNN",10)
    X_parent_cnn = np.expand_dims(X_parent_cnn,2)
    Y_parent_cnn = np.array([np.vstack((c_k,np.abs(1-c_k))).T for c_k in Y_parent_cnn])
    cnn_kf.perform_Kfold(X_parent_cnn,Y_parent_cnn,labels_codes,codes,subjects,subtest,prefix,4,0.25)


    # # Riemannian part
    X_parent,Y_parent,domains_parent = prepare_data(subjects,raw_data,labels,on_frame,True,recenter,codes=codes, normalise=False)

    # # SPD Kfolder
    print("Perform SPD\n")
    spd_kf = Kfolder("SPD",10)
    spd_kf.perform_Kfold(X_parent,Y_parent,labels_codes,codes,subjects,subtest,prefix,4,0.25)
    
    # # SPDBN Kfolder
    print("Perform SPDBN\n")
    spdbn_kf = Kfolder("SPDBN",10)
    spdbn_kf.perform_Kfold(X_parent,Y_parent,labels_codes,codes,subjects,subtest,prefix,4,0.25)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects",default='[1,2,3,4,5,6,7,8,9,10,11,12]', help="The index of the subject to get data from")
    parser.add_argument("--subtest",default=1,type=int, help="The index of the subject to test on")
    parser.add_argument("--on_frame",default=True,type=bool,help="Boolean to get epochs on frame wize or time sample wize")
    parser.add_argument("--recenter",default=True,type=bool,help="Boolean to recenter the data before classifying or not")

    args = parser.parse_args()

    main(args.subjects,args.subtest,args.on_frame,args.recenter)