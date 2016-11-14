import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle as pickle

def unpickle():
    with open('tune_params_per_model.pkl','r') as f:
        outputs = pickle.load(f)
    return outputs

def unpack(outputs):
    bost_out, diab_out = outputs
    wcv_scores = []
    final_scores = []
    for output in [bost_out, diab_out]:
        wcv_scores.append([tup[0] for tup in output])
        final_scores.append([tup[1] for tup in output])
    bost_data = wcv_scores[0],final_scores[0]
    diab_data = wcv_scores[1],final_scores[1]
    return bost_data, diab_data

def plot_data(data, title, colors=None):
    plt.figure()
    scaled_data_1 = np.array(data[1])/np.std(np.array(data[1]))
    if colors is not None:
        plt.scatter(data[0],scaled_data_1, color=colors)
    else:
        plt.scatter(data[0],scaled_data_1)
    plt.title(title)

def plot_all(outputs, colors=None):
    titles = ['Boston', 'Diabetes']
    for data, title in zip(unpack(outputs), titles):
        plot_data(data, title, colors=colors)
    plt.show()

if __name__ == '__main__':
    plot_all(unpickle())
