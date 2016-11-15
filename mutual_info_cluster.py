"""Find clustering with maximal mutual information"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mutual_info_score

class MaxMutualInfo(object):
    def __init__(self, k, max_iter=30):
        self.k = k
        self.max_iter = max_iter
        self.data = None
        self.weights = None
        self.assignments = None

    def fit(self, data):
        self.data, self.weights = data
        n_points = self.data.shape[0]
        self.assignments = np.random.randint(0,self.k,(n_points,))
        for _ in xrange(n_points*self.max_iter):
            point = np.random.randint(0,n_points)
            mutual_infos = []
            for cluster in xrange(self.k):
                possible_assignments = self.assignments.copy()
                possible_assignments[point] = cluster
                mutual_infos = np.apply_along_axis(lambda col: mutual_info_score(col, possible_assignments), 0, data)
                weighted_mean_mi = mutual_infos.dot(self.weights)
                mutual_infos.append(weighted_mean_mi)
            best = np.argmax(np.array(mutual_infos))
            self.assignments[point] = best
