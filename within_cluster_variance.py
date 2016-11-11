import numpy as np

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

class WCVScore(object):
    def __init__(self, model, sample=False):
        self.model = model
        self.wcvs = []
        self.sample = sample

    def score(self, data):
        self.wcvs = []
        n_features = data.shape[1]
        for i in xrange(n_features):
            if not self.sample == False:
                if np.random.random() > self.sample:
                    continue
            y = data[:,i]
            X = np.delete(data, i, axis=1)
            predictions = self.model.fit_predict(X)
            within_cluster_variance = mean_cluster_variances(predictions, y)
            total_variance = np.var(y)
            scaled_within_cluster_variance = within_cluster_variance / total_variance
            self.wcvs.append(scaled_within_cluster_variance)
        return np.mean(self.wcvs)
