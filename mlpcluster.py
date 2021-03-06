import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def fit_mlp(mlp, X, feature):
    y_temp = X[:,feature]
    X_temp = np.delete(X, feature, axis=1)
    return mlp.fit(X_temp,y_temp)

def transform_mlp(mlp, X, feature):
    y_temp = X[:,feature]
    X_temp = np.delete(X, feature, axis=1)
    return (mlp.predict(X_temp)).reshape(-1)

class MLPCluster(object):
    def __init__(self, hidden_layer_size=5, alpha=0.0001, training_fraction=1, standard_scaling=True, n_jobs=-1):
        self.hidden_layer_size = hidden_layer_size
        self.alpha = alpha
        self.training_fraction = training_fraction
        self.standard_scaling=standard_scaling
        self.n_jobs=n_jobs
        self.mlps = None
        self.pca = PCA()
        self.ss = StandardScaler()
        self.n_features = None

    def fit(self, data):
        if self.training_fraction < 1:
            n_rows = X.shape[0]
            shuffling = np.random.permutation(n_rows)
            X = X[shuffling]
            rows_to_keep = int(n_rows*self.training_fraction)
            data = data[:rows_to_keep]
        self.n_features = data.shape[1]
        if self.standard_scaling:
            X_ss = self.ss.fit_transform(data)
        else:
            X_ss = data
        X = self.pca.fit_transform(X_ss)
        self.mlps = [MLPRegressor(hidden_layer_sizes=(self.hidden_layer_size,), activation='logistic', alpha=self.alpha) for _ in xrange(self.n_features)]
        # for i in xrange(self.n_features):
        #     y_temp = X[:,i]
        #     X_temp = np.delete(X, i, axis=1)
        #     self.mlps[i].fit(X_temp,y_temp)
        self.mlps = Parallel(n_jobs=self.n_jobs)(delayed(fit_mlp)(mlp, X, i) for i, mlp in enumerate(self.mlps))
        return self

    def transform(self, data):
        if self.standard_scaling:
            X_ss = self.ss.transform(data)
        else:
            X_ss = data
        X = self.pca.transform(X_ss)
        # transformed_features = []
        # for i in xrange(self.n_features):
        #     X_temp = np.delete(X, i, axis=1)
        #     transformed_feature = self.mlps[i].predict(X_temp)
        #     transformed_features.append(transformed_feature.reshape(-1))
        transformed_features = Parallel(n_jobs=self.n_jobs)(delayed(transform_mlp)(mlp, X, i) for i, mlp in enumerate(self.mlps))
        transformed_data_pca = (np.array(transformed_features)).T
        transformed_data_ss = self.pca.inverse_transform(transformed_data_pca)
        if self.standard_scaling:
            transformed_data = self.ss.inverse_transform(transformed_data_ss)
        else:
            transformed_data = transformed_data_ss
        return transformed_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
