import numpy as np
from sklearn.datasets import load_boston, load_diabetes, load_iris, load_breast_cancer
from within_cluster_variance import WCVScore, DevariancedModel
from compare_algorithms import FullSLCluster, grid_search
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering, Birch
from joblib import Parallel, delayed
import cPickle as pickle

KMEANS_PARAMS = {
    'model': [KMeans],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'n_clusters': [2,3,4,5,6,7,8,9,10]
}

DBSCAN_PARAMS = {
    'model': [DBSCAN],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'eps': [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
}

AFFPROP_PARAMS = {
    'model': [AffinityPropagation],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'damping': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
}

AGGLOM_PARAMS = {
    'model': [AgglomerativeClustering],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'n_clusters': [2,3,4,5,6,7,8,9,10]
}

BIRCH_PARAMS = {
    'model': [Birch],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'n_clusters': [2,3,4,5,6,7,8,9,10]
}

SLC_PARAMS = {
    'model' : [FullSLCluster],
    'dataset' : ['boston','diabetes','iris','breast_cancer'],
    'k' : [2,3,4,5,6,8,10],
    'kmeans_type' : ['squishy','normal'],
    'n_forests' : [100],
    'n_trees' : [1],
    'n_features_to_predict' : [0.3,0.5],
    'max_depth' : [3,5,6],
    'weight_extent' : [0],
    'n_jobs' : [-1],
}

boston_data = load_boston().data
diabetes_data = load_diabetes().data
iris_data = load_iris().data
breast_cancer_data = load_breast_cancer().data

DATASETS = {
    'boston' : boston_data,
    'diabetes' : diabetes_data,
    'iris' : iris_data,
    'breast_cancer' : breast_cancer_data
}

def fit_model(params):
    dataset = params.pop('dataset')
    data = DATASETS[dataset]
    Model = params.pop('model')
    slc_model = Model(**params)
    model = DevariancedModel(slc_model)
    wcv = WCVScore(model)
    wcv_score, n_clusters = wcv.score(data)
    return dataset, params, wcv_score, n_clusters

def select_params_per_dataset_per_k(fit_model_output):
    datasets = [tup[0] for tup in fit_model_output]
    params = [tup[1] for tup in fit_model_output]
    ks = [param['k'] for param in params]
    wcv_scores = [tup[2] for tup in fit_model_output]
    unique_ks = np.unique(np.array(ks))
    unique_datasets = DATASETS.keys()
    output = []
    for dataset in unique_datasets:
        for k in unique_ks:
            mask = ((np.array(datasets)==dataset) & (np.array(ks)==k))
            min_wcv = np.argmin(np.array(wcv_scores)-10*mask)
            output.append((dataset,params[min_wcv]))
    return output

def prepare_tuned_models(select_params_output):
    params = []
    for row in select_params_output:
        param = row[1]
        param['dataset'] = row[0]
        param['model'] = FullSLCluster
        params.append(param)
    return params

if __name__ == '__main__':
    ## Make baseline data
    #baseline_models = ['dbscan', 'affprop', 'agglom', 'birch', 'kmeans']
    #baseline_params = [DBSCAN_PARAMS, AFFPROP_PARAMS, AGGLOM_PARAMS, BIRCH_PARAMS, KMEANS_PARAMS]
    baseline_models = ['affprop', 'agglom', 'birch', 'kmeans']
    baseline_params = [AFFPROP_PARAMS, AGGLOM_PARAMS, BIRCH_PARAMS, KMEANS_PARAMS]
    for model, params in zip(baseline_models, baseline_params):
        models_to_search = grid_search(params)
        fit_model_output = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in models_to_search)
        with open(model + '_baseline_models.pkl','w') as f:
            pickle.dump(fit_model_output, f)

    # ## Creating the RFCluster stuff
    # models_to_search = grid_search(SLC_PARAMS)
    # fit_model_output = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in models_to_search)
    # with open('big_ole_grid_search_output_no_pca.pkl','w') as f:
    #     pickle.dump(fit_model_output, f)
    # params_per_dataset = select_params_per_dataset_per_k(fit_model_output)
    # params_to_retry = prepare_tuned_models(params_per_dataset)
    # final_models = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in params_to_retry)
    # with open('final_models_from_grid_search_no_pca.pkl','w') as f:
    #     pickle.dump(final_models, f)
