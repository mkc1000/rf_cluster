import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances

class RFCluster(object):
    def __init__(self, n_forests, n_trees=1, n_features_to_predict=0.5, max_depth=5, outputting_weights=False, weight_extent=1):
        self.n_forests = n_forests
        self.n_trees = n_trees
        self.n_features_to_predict = n_features_to_predict
        self.outputting_weights = outputting_weights
        self.weight_extent = weight_extent
        self.rfs = [RandomForestRegressor(n_trees, max_depth=max_depth, n_jobs=-1) for _ in xrange(n_forests)]
        self.pca = PCA()
        self.ss1 = StandardScaler()
        self.decision_paths = None
        self.features_indices = []
        if outputting_weights:
            self.weights = []

    def fit(self, X_init):
        self.features_indices = []

        X_ss = self.ss1.fit_transform(X_init)
        X = self.pca.fit_transform(X_ss)
        if isinstance(self.n_features_to_predict, float):
            n_output = int(self.n_features_to_predict * X.shape[1])
        elif isinstance(self.n_features_to_predict, int):
            n_output = self.n_features_to_predict
        elif self.n_features_to_predict == 'sqrt':
            n_output = int(np.sqrt(X.shape[1]))
        elif self.n_features_to_predict == 'log':
            n_output = int(np.log2(X.shape[1]))

        for i in xrange(self.n_forests):
            features_to_predict = np.random.choice(np.arange(X.shape[1]),(n_output,),replace=False)
            self.features_indices.append(features_to_predict)
            y_temp = X[:, features_to_predict]
            X_temp = np.delete(X, features_to_predict, axis=1)
            self.rfs[i].fit(X_temp, y_temp)

    def transform(self, X_init):
        self.decision_paths = None
        if self.outputting_weights:
            self.weights = []

        X_ss = self.ss1.transform(X_init)
        X = self.pca.transform(X_ss)

        for i, features_to_predict in enumerate(self.features_indices):
            y_temp = X[:, features_to_predict]
            X_temp = np.delete(X, features_to_predict, axis=1)
            predictions = self.rfs[i].predict(X_temp)
            if self.outputting_weights:
                mean_squared_error = np.sum(np.square(y_temp - predictions))/y_temp.shape[0]
                y_temp_var = np.sum(np.apply_along_axis(np.var, 0, y_temp))
                var_reduction = (y_temp_var - mean_squared_error)/y_temp_var
                weight = var_reduction**self.weight_extent
                ##Alternative: based on varience of features to predict
                weight = y_temp_var**self.weight_extent
                self.weights.append(weight)
            if len(predictions.shape) > 1:
                predictions = np.sum(predictions, 1).reshape(-1,1)
            else:
                predictions = predictions.reshape(-1,1)
            if self.decision_paths is None:
                self.decision_paths = predictions
            else:
                self.decision_paths = np.hstack((self.decision_paths, predictions))

        if not self.outputting_weights:
            return self.decision_paths
        else:
            self.weights = np.array(self.weights)
            self.weights = self.weights/np.sum(self.weights)
            return self.decision_paths, self.weights

    def fit_transform(self, X_init, _=None):
        self.fit(X_init)
        return self.transform(X_init)

def jaccard(x,y):
    return np.mean(x!=y)

def weighted_jaccard(x,y,w):
    """w is list of weights, same length as x and y summing to 1"""
    return (x!=y).dot(w)

def jaccard_distance_matrix(X):
    vint = np.vectorize(int)
    X_int = vint(X*100)
    return pairwise_distances(X_int, metric=jaccard)

def weighted_jaccard_distance_matrix(X,w):
    """w has length X.shape[1]"""
    vint = np.vectorize(int)
    X_int = vint(X*100)
    wjaccard = lambda x,y: weighted_jaccard(x,y,w)
    return pairwise_distances(X_int, metric=wjaccard)

class JKMeans(object):
    def __init__(self, k, max_iter=None, n_attempts=10, accepting_weights=False):
        self.k = k
        self.n_attempts = n_attempts
        if max_iter is None:
            self.max_iter = -1
        else:
            self.max_iter = max_iter
        self.accepting_weights = accepting_weights
        self.distance_matrix = None
        self.assignments = None
        self.assignment_score = None

    def fit_once(self, X):
        assignments = np.random.randint(0,self.k,size=X.shape[0])
        old_assignments = np.zeros(assignments.shape)
        it = 0
        while (old_assignments != assignments).any() and it != self.max_iter:
            it += 1
            old_assignments = assignments
            centroids = []
            for cluster in xrange(self.k):
                mask = assignments == cluster
                if np.sum(mask) == 0:
                    continue
                within_cluster_distance_matrix = (self.distance_matrix[mask]).T
                most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
                centroids.append(most_central_point)
            to_centroid_distnace_matrix = (self.distance_matrix[centroids]).T
            assignments = np.apply_along_axis(np.argmin, 1, to_centroid_distnace_matrix)
        return assignments

    def score(self, assignments):
        centroids = []
        for cluster in xrange(self.k):
            mask = assignments == cluster
            if np.sum(mask) == 0:
                continue
            within_cluster_distance_matrix = (self.distance_matrix[mask]).T
            most_central_point = np.argmin(np.sum(within_cluster_distance_matrix,1))
            centroids.append(most_central_point)
        to_centroid_distnace_matrix = (self.distance_matrix[centroids]).T
        scores = np.apply_along_axis(np.min, 1, to_centroid_distnace_matrix)
        score = np.sum(scores)
        return score

    def fit(self, X):
        if self.accepting_weights:
            X, weights = X
            self.distance_matrix = weighted_jaccard_distance_matrix(X, weights)
        else:
            self.distance_matrix = jaccard_distance_matrix(X)
        for _ in xrange(self.n_attempts):
            assignments = self.fit_once(X)
            if self.assignments is None:
                self.assignments = assignments
                self.assignment_score = self.score(self.assignments)
            else:
                score = self.score(assignments)
                if score < self.assignment_score:
                    self.assignment_score = score
                    self.assignments = assignments
        return self

    def fit_predict(self, X, _=None):
        self.fit(X)
        return self.assignments
