import numpy as np
from slcluster import SLCluster, JKMeans, EigenvectorWeighting, SquishyJKMeans
from within_cluster_variance import WCVScore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, MeanShift, AffinityPropagation, SpectralClustering #, ward_tree
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
import itertools
import cPickle as pickle
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer, fetch_rcv1
from compare_algorithms import grid_search, score_model

class DoubleSLCluster(Pipeline):
    def __init__(self, k,
                model_type='random_forest',
                kmeans_type='squishy',
                n_forests=192,
                n_trees=1,
                n_features_to_predict=0.5,
                max_depth=5, #should be 2 for boosting
                learning_rate=0.6,
                using_weights=True,
                weight_extent=1, # 2 for boosting
                max_iter=60,
                n_attempts=10,
                weight_adjustment=0,
                eig_extent=0,
                n_jobs=1):
        slc1 = SLCluster(n_forests,
                        model_type=model_type,
                        n_trees=n_trees,
                        n_features_to_predict=n_features_to_predict,
                        max_depth=max_depth,
                        outputting_weights=False,
                        weight_extent=weight_extent,
                        learning_rate=learning_rate,
                        n_jobs=n_jobs)
        slc2 = SLCluster(n_forests,
                        model_type=model_type,
                        n_trees=n_trees,
                        n_features_to_predict=n_features_to_predict,
                        max_depth=max_depth,
                        outputting_weights=using_weights,
                        weight_extent=weight_extent,
                        learning_rate=learning_rate,
                        n_jobs=n_jobs)
        ew = EigenvectorWeighting(extent=eig_extent)
        if kmeans_type == 'normal':
            jk = JKMeans(k,
                            max_iter=max_iter,
                            n_attempts=n_attempts,
                            accepting_weights=using_weights,
                            weight_adjustment=weight_adjustment,
                            n_jobs=n_jobs)
        else:
            jk = SquishyJKMeans(k,
                            max_iter=max_iter,
                            n_attempts=n_attempts,
                            accepting_weights=using_weights,
                            weight_adjustment=weight_adjustment,
                            n_jobs=n_jobs)
        if eig_extent == 0:
            Pipeline.__init__(self,[('slc1', slc1), ('slc2', slc2), ('jkmeans', jk)])
        else:
            Pipeline.__init__(self,[('slc1', slc1), ('slc2', slc2), ('ew', ew), ('jkmeans', jk)])

MODEL_PARAMS = {
    'KMeans': {
        'model' : KMeans,
        'parameters' : {
            'n_clusters': [2,3,4,5,6,7]
        }
    },
    'DoubleSLCluster' : {
        'model' : DoubleSLCluster,
        'parameters' : {
            'k' : [3,5,7],
            'n_forests' : [150],
            'n_trees' : [1],
            'n_features_to_predict' : [0.5],
            'max_depth' : [4,5,6],
            'weight_extent' : [1,2],
            'n_jobs' : [-1]
        }
    }
}

def parameterized_models():
    for model_name, d in MODEL_PARAMS.iteritems():
        Model = d['model']
        params = d['parameters']
        for dictionary in grid_search(params):
            model = Model(**dictionary)
            yield model_name, model

if __name__ == '__main__':
    data = load_boston().data
    ss = StandardScaler()
    data_ss = ss.fit_transform(data)
    models = parameterized_models()
    output = Parallel(n_jobs=-1)(delayed(score_model)(model_name, model, data_ss) for model_name, model in models)
    with open('model_compare_double.pkl','w') as f:
        pickle.dump(output, f)
