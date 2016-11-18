import numpy as np
import itertools
import cPickle as pickle
from mlpcluster import MLPCluster
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1

DATASETS = {
    'boston' : load_boston().data,
    'diabetes' : load_diabetes().data,
    'iris' : load_iris().data,
    'breast_cancer' : load_breast_cancer().data,
    'rcv' : fetch_rcv1().data
}

if __name__ == '__main__':
    for name, data in DATASETS.iteritems():
        mlp = MLPCluster()
        transformed_data = mlp.fit_transform(data)
        filename = name + '_transformed.pkl'
        with open(filename, 'w') as f:
            pickle.dump(transformed_data, f)
