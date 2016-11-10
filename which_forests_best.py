import numpy as np
import cPickle as pickle
from collections import Counter
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

def look_up_in_dict(d, i):
    return d.get(i,0)

def turn_to_count_vector(n,vector):
    counter = Counter(vector)
    return np.array(Parallel(n_jobs=-1)(delayed(look_up_in_dict)(counter, i) for i in xrange(n)))

if __name__ == '__main__':
    with open('gs.pkl', 'r') as f:
        grid_search = pickle.load(f)
    y = grid_search[1]
    raw_X = grid_search[0]
    raw_X_ = zip(*raw_X)
    raw_X = np.array(raw_X_[1])
    n_cols = 600
    X = np.apply_along_axis(lambda row: turn_to_count_vector(n_cols, row), 1, raw_X)

    linreg = LinearRegression()
    linreg.fit(X, y)
    coefs = linreg.coef_

    with open('forest_relevances.pkl','w') as f:
        pickle.dump(coefs)
