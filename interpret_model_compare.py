import numpy as np
import cPickle as pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hists_feature(feat, X_series):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    ax1.hist(X_series[0][:,feat])
    ax2.hist(X_series[1][:,feat])
    ax3.hist(X_series[2][:,feat])
    ax4.hist(X_series[3][:,feat])
    plt.show()

if __name__ == '__main__':
    with open('model_compare.pkl','r') as f:
        models = pickle.load(f)
    with open('params_model_compare.pkl','r') as f:
        params = pickle.load(f)
    models, params = models[-720:], params[-720:]
    models = np.array([[tup[1],tup[2]] for tup in models])
    params_columns = params[0][1].keys() + ['n_clusters']
    params = np.array([tup[1].values() for tup in params])
    X = np.hstack((params,models[:,1].reshape(-1,1)))
    y = models[:,0]
    lr = LinearRegression()
    lr.fit(X[:,6].reshape(-1,1),y)
    y_pred = lr.predict(X[:,6].reshape(-1,1))
    y_resid = y - y_pred
    std_y_resid = np.std(y_resid)
    best_Xs = X[y_resid < -std_y_resid]
    bestbest_Xs = X[y_resid < -1.5*std_y_resid]
    bestbestbest_Xs = X[y_resid < -2*std_y_resid]
    X_series = (X, best_Xs, bestbest_Xs, bestbestbest_Xs)
