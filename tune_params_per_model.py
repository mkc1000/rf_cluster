import numpy as np
import cPickle as pickle
from compare_algorithms import grid_search, FullSLCluster
from within_cluster_variance import WCVScore, mean_cluster_variances
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1
from joblib import Parallel, delayed
import random
from datetime import datetime

SLC_PARAMS = {
        'k' : [4,6,7],
        'n_forests' : [150],
        'n_trees' : [1],
        'n_features_to_predict' : [0.5],
        'max_depth' : [3,5,8,10],
        'weight_extent' : [1,1.5,2,2.5],
        'eig_extent' : [0,3,6,10]
    }

def test_params(param_dict, data):
    random.seed(datetime.now())
    np.random.seed(random.randint(0,100000))
    print "Testing randomizer: ", np.random.choice(np.arange(10))
    print "starting to test ", param_dict
    features_to_withhold = np.random.choice(np.arange(data.shape[1]),2)
    y = data[:,features_to_withhold]
    data_limited = np.delete(data,features_to_withhold,axis=1)
    slc = FullSLCluster(**param_dict)
    wcv = WCVScore(slc)
    wcv_score, _ = wcv.score(data_limited)
    assignments = slc.fit_predict(data_limited)
    final_scores = np.apply_along_axis(lambda col: mean_cluster_variances(assignments, col), 0, y)
    final_score = np.mean(final_scores)
    print "done testing ", param_dict
    return wcv_score, final_score, features_to_withhold

if __name__ == '__main__':
    params_list = grid_search(SLC_PARAMS)
    datasets = [load_boston().data, load_diabetes().data, load_breast_cancer().data]
    outputs = []
    for data in datasets:
        outputs.append( Parallel(n_jobs=-1)(delayed(test_params)(param_dict, data) for param_dict in params_list) )
    with open('big_tune_params_by_model.pkl','w') as f:
        pickle.dump(outputs, f)
    with open('big_tune_params.pkl','w') as f:
        pickle.dump(params_list,f)
