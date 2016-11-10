import numpy as np
from rfcluster import RFCluster, JKMeans
from within_cluster_variance import WCVScore, mean_cluster_variances
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from joblib import Parallel, delayed
import cPickle as pickle

boston = load_boston()
data = boston.data
target = boston.target

def cross_product_2(lst1, lst2):
    """lst1 is a list of tuples"""
    output = []
    for item1 in lst1:
        for item2 in lst2:
            lst_item1 = list(item1)
            lst_item1.append(item2)
            output.append(tuple(lst_item1))
    return output

def cross_product(lists):
    lists_w_empty_tuple = [[tuple(())]] + lists
    return reduce(cross_product_2, lists_w_empty_tuple)

def apply_over_multiple_lists(func, lists):
    """func takes one argument: a tuple of length however many lists there are"""
    flattened = cross_product(lists)
    output = Parallel(n_jobs=-1)(delayed(func)(tup) for tup in flattened)
    return flattened, output

n_forests = 600

ks = range(2,8)
forests_to_use = [np.random.choice(np.arange(n_forests), size=n_forests/2) for _ in range(10)]

def score(data_t, forests_to_use, k):
    data = data_t[:,forests_to_use]
    jk = JKMeans(k, n_attempts=30)
    assignments = jk.fit_predict(data)
    mcv = mean_cluster_variances(assignments, target)
    total_variance = np.var(target)
    scaled_within_cluster_variance = mcv / total_variance
    return scaled_within_cluster_variance

if __name__ == '__main__':
    rfc = RFCluster(n_forests,1,0.5,5)
    data_t = rfc.fit_transform(data)
    with open('rfc.pkl', 'w') as f:
        pickle.dump(rfc, f)

    def score_lite((k, forests_to_use)):
        return score(data_t, forests_to_use, k)

    grid_search = apply_over_multiple_lists(score_lite, [ks, forests_to_use])
    with open('gs.pkl', 'w') as f:
        pickle.dump(grid_search, f)



    # tranformations = Parallel(n_jobs=-1)(delayed(transformations)(feature_withholding) for feature_withholding in features_withholding)
