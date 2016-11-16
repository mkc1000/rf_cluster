from within_cluster_variance import DevariancedModel
from compare_algorithms import FullSLCluster
from sklearn.datasets import load_boston
import cPickle as pickle
import numpy as np

if __name__ == '__main__':
    slc_wcvs = []
    dvm_wcvs = []
    boston = load_boston()
    data = boston.data
    target = boston.target
    for _ in xrange(4):
        slc = FullSLCluster(5, n_jobs=-1)
        dvm = DevariancedModel(slc)
        assignments_0 = slc.fit_predict(data)
        assignments_1 = dvm.fit_predict(data)
        wcv_0 = np.sum([np.var(target[assignments_0 == i])*np.mean([assignments_0 == i]) for i in np.unique(assignments_0)])/np.var(target)
        wcv_1 = np.sum([np.var(target[assignments_1 == i])*np.mean([assignments_1 == i]) for i in np.unique(assignments_1)])/np.var(target)
        slc_wcvs.append(wcv_0)
        dvm_wcvs.append(wcv_1)
    output = [slc_wcvs,dvm_wcvs]
    with open('test_dvm.pkl','w') as f:
        pickle.dump(output,f)
