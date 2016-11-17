import numpy as np
from mlpcluster import MLPCluster
import cPickle as pickle
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

def load_mnist():
    with open('mnist.pkl','r') as f:
        mnist = pickle.load(f)
    X = mnist.data
    numerals = mnist.target
    return X, numerals

if __name__ == '__main__':
    X, numerals = load_mnist()
    mlpc = MLPCluster(hidden_layer_size=60)
    print "starting to transform"
    X_transform = mlpc.fit_transform(X)
    print "done transforming"
    kmeans = KMeans(n_clusters=5)
    pred0 = kmeans.fit_predict(X)
    pred1 = kmeans.fit_predict(X_transform)
    mutinfo0 = mutual_info_score(pred0,numerals)
    mutinfo1 = mutual_info_score(pred1,numerals)
    output = pred0, pred1, numerals, mutinfo0, mutinfo1
    with open('mlp_compare.pkl','w') as f:
        pickle.dump(output, f)
