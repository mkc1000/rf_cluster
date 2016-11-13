import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns

def plot(filename):
    with open(filename,'r') as f:
        models = pickle.load(f)
    model_scatter = np.array([(tup[1],tup[2]) for tup in models if tup[2] < 30])
    color_d = {'AA_SLCluster': 'b', 'KMeans' : 'g','DBSCAN':'y','AffinityPropegation':'r'}
    colors = [color_d[tup[0]] for tup in models if tup[2] < 30]
    plt.scatter(model_scatter[:,1],model_scatter[:,0], color=colors)
    plt.show()


if __name__ == '__main__':
    filename = 'small_model_compare.pkl'
    plot(filename)
