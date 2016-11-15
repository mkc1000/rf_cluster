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
                all_mutual_infos = np.apply_along_axis(lambda col: mutual_info_score(col, possible_assignments), 0, self.data)
                weighted_mean_mi = all_mutual_infos.dot(self.weights)
                mutual_infos.append(weighted_mean_mi)
            best = np.argmax(np.array(mutual_infos))
            self.assignments[point] = best

class VotingCombination(object):
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
            def vote(col,point,assignments):
                votes = assignments[col==col[point]]
                vote = np.argmax(np.array([np.sum(votes==i) for i in xrange(self.k)]))
                return vote
            votes = np.apply_along_axis(lambda col: vote(col,point,self.assignments),0,self.data)
            vote_counts = [0 for _ in xrange(self.k)]
            for vote, weight in zip(votes, self.weights):
                vote_counts[vote] += weight
            self.assignments[point] = np.argmax(vote_counts)
