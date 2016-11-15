import numpy as np
from sklearn.datasets import fetch_mldata
from compare_algorithms import FullSLCluster

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    numerals = mnist.target
    slc = FullSLCluster(k=10,
                model_type='gradient_boosting',
                n_forests=150,
                n_trees=2,
                n_features_to_predict=0.5,
                max_depth=2, #should be 2 for boosting
                learning_rate=0.6,
                using_weights=True,
                weight_extent=2, # 2 for boosting
                max_iter=60,
                n_attempts=10,
                weight_adjustment=0,
                eig_extent=5,
                n_jobs=-1)
    assignments = slc.fit_predict(X)
