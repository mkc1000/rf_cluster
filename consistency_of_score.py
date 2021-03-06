import numpy as np
import cPickle as pickle
from compare_algorithms import grid_search, FullSLCluster
from within_cluster_variance import WCVScore, mean_cluster_variances
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1
from joblib import Parallel, delayed
import random
from datetime import datetime

SLC_PARAMS = {
        'k' : [3,5,7],
        'n_forests' : [600,300],
        'max_depth' : [3,5],
        'weight_extent' : [1,1.5,2],
        'kmeans_type' : ['squishy'],
        'learning_rate' : [0,1] #this parameter is not even used
    }

def test_params(param_dict, data):
    random.seed(datetime.now())
    np.random.seed(random.randint(0,100000))
    print "starting to test ", param_dict
    slc1 = FullSLCluster(**param_dict)
    wcv1 = WCVScore(slc1)
    wcv_score1, _ = wcv1.score(data)
    print "done testing ", param_dict
    return wcv_score1

if __name__ == '__main__':
    params_list = grid_search(SLC_PARAMS)
    data = load_diabetes().data
    output = Parallel(n_jobs=-1)(delayed(test_params)(param_dict, data) for param_dict in params_list)
    dummy_division = np.array([d['learning_rate'] for d in params_list])
    output1 = np.array(output)[dummy_division==1]
    output2 = np.array(output)[dummy_division!=1]
