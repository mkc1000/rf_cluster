import numpy as np
import matplotlib.pyplot as plt
#rimport seaborn as sns
import cPickle as pickle

def load_model_data():
    with open('final_models_from_grid_search_no_pca.pkl','r') as f:
        model_data = pickle.load(f)
    return model_data

def load_kmeans_data():
    with open('kmeans_benchmark.pkl','r') as f:
        kmeans_data = pickle.load(f)
    return kmeans_data

def plot_performance(model_data, kmeans_data):
    d = {'boston': 'Boston Housing', 'breast_cancer' : 'Breast Cancer', 'diabetes' : 'Diabetes', 'iris' : 'Iris'}
    datasets = np.unique([model[0] for model in model_data])
    fig = plt.figure(figsize=(10,6))
    plt.subplots_adjust(hspace=0.5) #,top=0.85)
    for i, dataset in enumerate(datasets):
        k_data_rfc = [model[3] for model in model_data if model[0] == dataset]
        wcv_data_rfc = [model[2] for model in model_data if model[0] == dataset]
        k_data_kmeans = [model[3] for model in kmeans_data if model[0] == dataset]
        wcv_data_kmeans = [model[2] for model in kmeans_data if model[0] == dataset]
        ax = fig.add_subplot(2,2,i+1)
        ax.scatter(k_data_kmeans, wcv_data_kmeans, color='r', marker='^', label='k-Means')
        ax.scatter(k_data_rfc, wcv_data_rfc, color='b', label='IRFC')
        if i == 3:
            plt.legend(loc=(-0.28,1.07),fontsize=11)
        plt.title(d[dataset])
    # bigax = fig.add_subplot(1,1,1,frameon=False)
    # bigax.axis('off')
    plt.text(-2.4,-0.06,'Number of Clusters', size=14)
    plt.text(-13,1.1,'Figure of Merit Score', size=15, rotation=90)


if __name__ == '__main__':
    model_data = load_model_data()
    kmeans_data = load_kmeans_data()
