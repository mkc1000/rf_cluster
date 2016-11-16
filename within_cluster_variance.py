import numpy as np
from joblib import Parallel, delayed
import random
from datetime import datetime

def mean_cluster_variances(clusters, feature):
    unique_clusters = np.unique(clusters)
    variances = []
    sizes = []
    for cluster in unique_clusters:
        cluster_feature = feature[clusters==cluster]
        variances.append(np.var(cluster_feature))
        sizes.append(len(cluster_feature))
    return np.array(variances).dot(np.array(sizes))/len(feature)

"""
At some point, try making an object that can tune hyperparameters (of RFCluster and JKMeans)
for a given dataset, by withholding a feature one at a time, running the whole model,
and looking at wcv on that feature.
"""

def score_once(wcv, data, i):
    y = data[:,i]
    X = np.delete(data, i, axis=1)
    predictions = wcv.model.fit_predict(X)
    n_clusters = len(np.unique(predictions))
    within_cluster_variance = mean_cluster_variances(predictions, y)
    total_variance = np.var(y)
    scaled_within_cluster_variance = within_cluster_variance / total_variance
    return scaled_within_cluster_variance, n_clusters

class WCVScore(object):
    def __init__(self, model, max_iter=10, n_jobs=1):
        self.model = model
        self.wcvs = []
        self.n_clusters = []
        self.sample = sample
        self.n_jobs=n_jobs
        self.max_iter = max_iter

    def score(self, data):
        self.wcvs = []
        n_features = data.shape[1]
        features = np.arange(n_features)
        if n_features > self.max_iter:
            features = np.random.choice(features,size=self.max_iter,replace=False)
        output = Parallel(n_jobs=self.n_jobs)(delayed(score_once)(self, data, i) for i in features)
        output = np.array(output)
        self.wcvs = output[:,0]
        self.n_clusters = output[:,1]
        return np.mean(self.wcvs), np.mean(self.n_clusters)

# class WCVScore(object):
#     def __init__(self, model, sample=False, n_jobs=1):
#         self.model = model
#         self.wcvs = []
#         self.n_clusters = []
#         self.sample = sample
#
#     def score(self, data):
#         self.wcvs = []
#         n_features = data.shape[1]
#         for i in xrange(n_features):
#             if not self.sample == False:
#                 if np.random.random() > self.sample:
#                     continue
#             y = data[:,i]
#             X = np.delete(data, i, axis=1)
#             predictions = self.model.fit_predict(X)
#             self.n_clusters.append(len(np.unique(predictions)))
#             within_cluster_variance = mean_cluster_variances(predictions, y)
#             total_variance = np.var(y)
#             scaled_within_cluster_variance = within_cluster_variance / total_variance
#             self.wcvs.append(scaled_within_cluster_variance)
#         return np.mean(self.wcvs), np.mean(self.n_clusters)

from sklearn.metrics import mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances

class DevariancedModel(object):
    def __init__(self,model,n_attempts=4):
        self.model=model
        self.n_attempts=n_attempts
        self.prediction = None

    def fit_once(self, data):
        random.seed(datetime.now())
        np.random.seed(random.randint(0,100000))
        all_predictions = []
        wcvs = []
        for i in xrange(data.shape[1]):
            y = data[:,i]
            X = np.delete(data, i, axis=1)
            predictions = self.model.fit_predict(X)
            all_predictions.append(predictions)
            within_cluster_variance = mean_cluster_variances(predictions, y)
            total_variance = np.var(y)
            scaled_within_cluster_variance = within_cluster_variance / total_variance
            wcvs.append(scaled_within_cluster_variance)
        wcv = np.mean(np.array(wcvs))
        all_predictions = (np.array(all_predictions)).T
        return wcv, all_predictions

    def fit(self,data):
        print 'starting to fit dvm'
        wcvs = []
        all_all_predictions = []
        for _ in xrange(self.n_attempts):
            wcv, all_predictions = self.fit_once(data)
            wcvs.append(wcv)
            all_all_predictions.append(all_predictions)
        best = np.argmax(np.array(wcvs))
        all_predictions = all_all_predictions[best]
        mutual_info_distance_matrix = pairwise_distances(all_predictions.T, metric=mutual_info_score)
        best_prediction = all_predictions[:,np.argmax(np.sum(mutual_info_distance_matrix,0))]
        self.prediction = best_prediction
        print 'done fitting dvm'
        return self

    def predict(self,data):
        return self.prediction

    def fit_predict(self,data):
        self.fit(data)
        return self.prediction
