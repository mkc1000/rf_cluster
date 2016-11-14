import numpy as np
from slcluster import SLCluster, JKMeans, EigenvectorWeighting
from within_cluster_variance import WCVScore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, MeanShift, AffinityPropagation, SpectralClustering #, ward_tree
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import itertools
import cPickle as pickle
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1
#from sklearn.datasets.mldata import fetch_mldata

class FullSLCluster(Pipeline):
    def __init__(self, k,
                n_forests=150,
                n_trees=1,
                n_features_to_predict=0.5,
                max_depth=5,
                using_weights=True,
                weight_extent=1,
                max_iter=None,
                n_attempts=10,
                weight_adjustment=0,
                eig_extent=0):
        slc = SLCluster(n_forests,
                        n_trees=n_trees,
                        n_features_to_predict=n_features_to_predict,
                        max_depth=max_depth,
                        outputting_weights=using_weights,
                        weight_extent=weight_extent)
        ew = EigenvectorWeighting(extent=eig_extent)
        jk = JKMeans(k,
                        max_iter=max_iter,
                        n_attempts=n_attempts,
                        accepting_weights=using_weights,
                        weight_adjustment=weight_adjustment)
        if eig_extent == 0:
            Pipeline.__init__(self,[('slc', slc), ('jkmeans', jk)])
        else:
            Pipeline.__init__(self,[('slc', slc), ('ew', ew), ('jkmeans', jk)])


MODEL_PARAMS = {
    # 'DBSCAN': {
    #     'model' : DBSCAN,
    #     'parameters' : {
    #         'eps' : [0.1,0.3,0.5,0.7],
    #         'min_samples' : [2,4,6,8,10]
    #     }
    # },
    'KMeans': {
        'model' : KMeans,
        'parameters' : {
            'n_clusters': [2,3,4,5,6,7]
        }
    },
    # 'MeanShift': {
    #     'model' : MeanShift,
    #     'parameters' : {}
    # },
    # 'AffinityPropegation': {
    #     'model' : AffinityPropagation,
    #     'parameters' : {
    #         'damping' : [0.75,0.76,0.77,0.78,0.79,0.8,0.825,0.85,0.9],
    #         'convergence_iter' : [1,2]
    #     }
    # },
    # 'SpectralClustering': {
    #     'model' : SpectralClustering,
    #     'parameters' : {
    #         'n_clusters' : [2,3,4,5,6,7,8]
    #     }
    # },
    # 'AA_SLCluster' : {
    #     'model' : FullSLCluster,
    #     'parameters' : {
    #         'k' : [2,3,4,5,6,7,8,9,10],
    #         'n_forests' : [150],
    #         'n_trees' : [1,2],
    #         'n_features_to_predict' : [0.3,0.5],
    #         'max_depth' : [3,4,5,6],
    #         'weight_extent' : [0.75,1,1.5,2,2.5]
    #     }
    'SLCluster' : {
        'model' : FullSLCluster,
        'parameters' : {
            'k' : [3,7],
            'n_forests' : [100,200],
            'n_trees' : [1],
            'n_features_to_predict' : [0.5],
            'max_depth' : [3],
            'weight_extent' : [1.5],
            'eig_extent': [0,1,2,4]
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
    data = load_breast_cancer().data
    ss = StandardScaler()
    data_ss = ss.fit_transform(data)
    models = parameterized_models()
    output = Parallel(n_jobs=-1)(delayed(score_model)(model_name, model, data_ss) for model_name, model in models)
    with open('model_compare_w_eig.pkl','w') as f:
        pickle.dump(output, f)
