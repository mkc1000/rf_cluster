import numpy as np
from sklearn.datasets import fetch_covtype, fetch_20newsgroups_vectorized, fetch_california_housing
from within_cluster_variance import WCVScore
from compare_algorithms import FullSLCluster, grid_search
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import cPickle as pickle
from sklearn.decomposition import PCA

KMEANS_PARAMS = {
    'model': [KMeans],
    'dataset' : ['covtype','news','california'],
    'n_clusters': [2,3,4,5,6,7,8,9,10]
}

SLC_PARAMS = {
    'model' : [FullSLCluster],
    'dataset' : ['covtype','news','california'],
    'k' : [2,3,4,5,6,8,10],
    'using_pca' : [False],
    'kmeans_type' : ['squishy'],
    'n_forests' : [128],
    'n_trees' : [1],
    'n_features_to_predict' : [0.5],
    'max_depth' : [3,5,6],
    'weight_extent' : [0],
    'n_jobs' : [-1],
}

covtype_data = fetch_covtype().data
news_data = fetch_20newsgroups_vectorized().data
news_pca = PCA(40)
news_data = news_pca.fit_transform(news_data.todense())
california_data = fetch_california_housing().data

DATASETS = {
    'covtype' : covtype_data,
    'news' : news_data,
    'california' : california_data,
}

def fit_model(params):
    dataset = params.pop('dataset')
    data = DATASETS[dataset]
    Model = params.pop('model')
    slc_model = Model(**params)
    wcv = WCVScore(slc_model)
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
    models_to_search = grid_search(SLC_PARAMS)
    # fit_model_output = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in models_to_search)
    fit_model_output = [fit_model(model) for model in models_to_search]
    with open('big_ole_grid_search_new_datasets.pkl','w') as f:
        pickle.dump(fit_model_output, f)
    params_per_dataset = select_params_per_dataset_per_k(fit_model_output)
    params_to_retry = prepare_tuned_models(params_per_dataset)
    # final_models = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in params_to_retry)
    final_models = [fit_model(model) for model in params_to_retry]
    with open('final_models_from_grid_search_new_datasets.pkl','w') as f:
        pickle.dump(final_models, f)

    models_to_search = grid_search(KMEANS_PARAMS)
    # fit_model_output = Parallel(n_jobs=-1)(delayed(fit_model)(model) for model in models_to_search)
    fit_model_output = [fit_model(model) for model in models_to_search]
    with open('big_ole_grid_search_new_datasets_kmeans.pkl','w') as f:
        pickle.dump(fit_model_output, f)
