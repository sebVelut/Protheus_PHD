import numpy as np
from scipy import stats
from pyriemann.utils import distance
from pyriemann.utils.mean import mean_covariance

from Alignments.covariance import compute_covariances

def get_interrested_corr(corr_stat):
    length = corr_stat.shape[0]
    return np.array([corr_stat[i,length//2+i] for i in range(length//2)])

def spearman_correlation_similarity(Xt, Yt, Xs, Ys):
    corr_1 = stats.spearmanr(np.mean(Xs[Ys==1],axis=0),np.mean(Xt[Yt==1],axis=0),axis=1)
    corr_0 = stats.spearmanr(np.mean(Xs[Ys==0],axis=0),np.mean(Xt[Yt==0],axis=0),axis=1)

    return [get_interrested_corr(corr_1.statistic), get_interrested_corr(corr_0.statistic)]

def pearson_correlation_similarity(Xt, Yt, Xs, Ys):
    corr_1 = np.array([stats.pearsonr(np.mean(Xs[Ys==1][i],axis=0),np.mean(Xt[Yt==1][i],axis=0)) for i in range(Xs.shape[1])])
    corr_0 = np.array([stats.pearsonr(np.mean(Xs[Ys==0][i],axis=0),np.mean(Xt[Yt==0][i],axis=0)) for i in range(Xs.shape[1])])

    return [corr_1, corr_0]

def cosine_similarity(Xt, Yt, Xs, Ys):
    Xt_1_mean = np.mean(Xt[Yt==1],axis=0).reshape(Xt.shape[1],Xt.shape[2])
    Xs_1_mean = np.mean(Xs[Ys==1],axis=0).reshape(Xs.shape[1],Xt.shape[2])
    num = np.dot(Xt_1_mean,np.transpose(Xs_1_mean))
    cos_sim_1 = num/np.linalg.norm(Xt_1_mean,axis=1)*np.linalg.norm(Xs_1_mean,axis=1)
    
    Xt_0_mean = np.mean(Xt[Yt==0],axis=0).reshape(Xt.shape[1],Xt.shape[2])
    Xs_0_mean = np.mean(Xs[Ys==0],axis=0).reshape(Xs.shape[1],Xt.shape[2])
    num = np.dot(Xt_0_mean,np.transpose(Xs_0_mean))
    cos_sim_0 = num/np.linalg.norm(Xt_0_mean,axis=1)*np.linalg.norm(Xs_0_mean,axis=1)

    return [cos_sim_1,cos_sim_0]

def riemannian_distance(Xt, Yt, Xs, Ys):
    # Convert matrices into SPD matrices
    covt_1 = mean_covariance(compute_covariances(Xt[Yt==1], estimator='lwf'), metric='riemann')
    covt_0 = mean_covariance(compute_covariances(Xt[Yt==0], estimator='lwf'), metric='riemann')

    covs_1 = mean_covariance(compute_covariances(Xs[Ys==1], estimator='lwf'), metric='riemann')
    covs_0 = mean_covariance(compute_covariances(Xs[Ys==0], estimator='lwf'), metric='riemann')
    # Calculate distance

    dist_0 = distance.distance(covt_0,covs_0)
    dist_1 = distance.distance(covt_1,covs_1)

    return [1/np.exp(dist_1), 1/np.exp(dist_0)]

def euclidian_distance(Xt, Yt, Xs, Ys):

    dist1 = np.linalg.norm(np.mean(Xt[Yt==1],axis=0) - np.mean(Xs[Ys==1],axis=0))
    dist0 = np.linalg.norm(np.mean(Xt[Yt==0],axis=0) - np.mean(Xs[Ys==0],axis=0))

    return [1/np.exp(dist1), 1/np.exp(dist0)]