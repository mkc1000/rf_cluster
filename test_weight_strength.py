import numpy as np
from slcluster import SLCluster, JKMeans
from within_cluster_variance import WCVScore
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston, load_diabetes
from joblib import Parallel, delayed

boston = load_boston()
data = boston.data

diabetes = load_diabetes()
data = diabetes.data

def score_weight(weight):
    slc = Pipeline([('slc', SLCluster(n_forests=60,n_trees=1,n_features_to_predict=0.5,max_depth=5,outputting_weights=True,weight_extent=weight)), ('jkmeans', JKMeans(5, n_attempts=6,accepting_weights=True))])
    wcv = WCVScore(slc)
    print "Starting to score model with weight", weight
    score = wcv.score(data)
    print weight
    print score
    return score

if __name__ == '__main__':
    output = Parallel(n_jobs=-1)(delayed(score_weight)(weight/20.) for weight in range(64))
