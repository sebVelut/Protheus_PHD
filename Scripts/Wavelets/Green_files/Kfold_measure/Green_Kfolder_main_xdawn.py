import numpy as np
from Green_Kfold_measure_xdawn import Green_Kfolder_Xdawn
from utils import get_BVEP_data, prepare_data
import sys
import argparse
from pyriemann.estimation import Xdawn
import os
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\datasets")
sys.path.insert(0,"C:\\Users\\s.velut\\Documents\\These\\moabb\\moabb\\paradigms")
from castillos2023 import CasitllosCVEP100,CasitllosCVEP40,CasitllosBurstVEP100,CasitllosBurstVEP40


def main(subjects,subtest,on_frame,recenter,window_size):
    if not os.path.exists("./results/score/"):
        os.makedirs("./results/score/")
    if not os.path.exists("./results/score_code/"):
        os.makedirs("./results/score_code/")
    if not os.path.exists("./results/tps_train/"):
        os.makedirs("./results/tps_train/")
    if not os.path.exists("./results/tps_test/"):
        os.makedirs("./results/tps_test/")

    subjects = eval(subjects)
    recenter = eval(recenter)
    on_frame = eval(on_frame)
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

    moabb_ds = CasitllosBurstVEP100(window_size)

    raw_data,labels,codes,labels_codes = get_BVEP_data(subjects,on_frame,to_keep=None,moabb_ds=moabb_ds,window_size=window_size)
    X_parent,Y_parent,domains_parent = prepare_data(subjects,raw_data,labels,on_frame,False,recenter,codes,window_size=window_size)

    X = np.zeros((X_parent.shape[0],X_parent.shape[1],X_parent.shape[2],int(window_size*500)))
    for i in range(len(subjects)):
        xdawn = Xdawn(nfilter=16,classes=[1],estimator='lwf')
        temp_X = xdawn.fit_transform(X_parent[i],Y_parent[i])
        X[i] = np.hstack([temp_X,np.tile(xdawn.evokeds_[None,:,:],(temp_X.shape[0],1,1))])
    

    # CNN Kfolder
    print("Perform Grenn\n")
    kf = Green_Kfolder_Xdawn(10)
    kf.perform_Kfold(X,Y_parent,domains_parent,labels_codes,codes,subjects,subtest,prefix,4,window_size)









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects",default='[1,2,3,4,5,6,7,8,9,10,11,12]', help="The index of the subject to get data from")
    parser.add_argument("--subtest",default=1,type=int, help="The index of the subject to test on")
    parser.add_argument("--on_frame",default=True,help="Boolean to get epochs on frame wize or time sample wize")
    parser.add_argument("--recenter",default=True,help="Boolean to recenter the data before classifying or not")
    parser.add_argument("--ws",default=0.25,type=float,help="Boolean to recenter the data before classifying or not")

    args = parser.parse_args()

    main(args.subjects,args.subtest,args.on_frame,args.recenter,args.ws)

