import numpy as np
from sklearn.datasets import fetch_mldata
from compare_algorithms import FullSLCluster
import cPickle as pickle

if __name__ == '__main__':
    with open('mnist.pkl','r') as f:
        mnist = pickle.load(f)
    X = mnist.data
    numerals = mnist.target
    slc = FullSLCluster(k=5,
                model_type='random_forest',
                kmeans_type='squishy'
                n_forests=192,
                n_trees=2,
                n_features_to_predict=0.4,
                max_depth=5, #should be 2 for boosting
                learning_rate=0.6,
                using_weights=True,
                weight_extent=1.5, # 2 for boosting
                max_iter=60,
                n_attempts=10,
                weight_adjustment=0,
                eig_extent=1,
                n_jobs=-1)
    assignments = slc.fit_predict(X)
