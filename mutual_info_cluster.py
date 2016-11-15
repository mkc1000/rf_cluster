"""Find clustering with maximal mutual information"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mutual_info_score

class MaxMutualInfo(object):
    def __init__(self, k):
        self.k = k
        self.data = None
        self.weights = None
        self.assignments = None

    def fit(self, data):
        self.data, self.weights = data
