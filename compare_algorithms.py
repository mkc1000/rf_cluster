import numpy as np
from rfcluster import RFCluster, JKMeans
from within_cluster_variance import WCVScore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, MeanShift, AffinityPropagation, SpectralClustering #, ward_tree
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import itertools
import cPickle as pickle
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1
#from sklearn.datasets.mldata import fetch_mldata

MODEL_PARAMS = {
    'DBSCAN': {
        'model' : DBSCAN,
        'parameters' : {
            'eps' : [0.1,0.7],
            'min_samples' : [2,10]
        }
    },
    'KMeans': {
        'model' : KMeans,
        'parameters' : {
            'n_clusters': [2,12]
        }
    },
    'MeanShift': {
        'model' : MeanShift,
        'parameters' : {}
    },
    'AffinityPropegation': {
        'model' : AffinityPropagation,
        'parameters' : {
            'damping' : [0.5,0.9],
            'convergence_iter' : [3,15]
        }
    },
    'SpectralClustering': {
        'model' : SpectralClustering,
        'parameters' : {
            'n_clusters' : [2,12]
        }
    }
}

def grid_search(d):
    "Takes dictionary, where each value is a list. Returns list of dictionaries, where all possible selections from lists are the values."
    if len(d.keys()) == 0:
        return [{}]
    if len(d.keys()) == 1:
        values = d.values()[0]
        return [{d.keys()[0]: value} for value in values]
    output = []
    d_copy = d.copy()
    first_key = d.keys()[0]
    first_list = d_copy.pop(first_key)
    reduced_grid_search = grid_search(d_copy)
    for value in first_list:
        for d in reduced_grid_search:
            d_copy2 = d.copy()
            d_copy2[first_key] = value
            output.append(d_copy2)
    return output

def parameterized_models():
    for model_name, d in MODEL_PARAMS.iteritems():
        Model = d['model']
        params = d['parameters']
        for dictionary in grid_search(params):
            model = Model(**dictionary)
            yield model_name, model

def score_model(model_name, model, data):
    print "Starting to score " + model_name
    wcv = WCVScore(model)
    score, n_clusters = wcv.score(data)
    print "Done scoring " + model_name
    return model_name, score, n_clusters

if __name__ == '__main__':
    data = load_boston().data
    ss = StandardScaler()
    data_ss = ss.fit_transform(data)
    models = parameterized_models()
    output = Parallel(n_jobs=-1)(delayed(score_model)(model_name, model, data_ss) for model_name, model in models)
    with open('model_compare.pkl','w') as f:
        pickle.dump(output, f)
