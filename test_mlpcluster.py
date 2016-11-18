import numpy as np
from mlpcluster import MLPCluster
import cPickle as pickle
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score

def load_mnist():
    with open('mnist.pkl','r') as f:
        mnist = pickle.load(f)
    X = mnist.data
    numerals = mnist.target
    return X, numerals

def visualize_mnist_row(row):
    import matplotlib.pyplot as plt #Sorry about the awkward position for the import; EC2 instances can't import this
    pixels = np.array(row, dtype='uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.imshow(pixels, cmap='gray')
    plt.show()

if __name__ == '__main__':
    X, numerals = load_mnist()
    mlpc = MLPCluster(hidden_layer_size=5)
    print "starting to transform"
    X_transform = mlpc.fit_transform(X)
    print "done transforming"
    kmeans = KMeans(n_clusters=5)
    pred0 = kmeans.fit_predict(X)
    pred1 = kmeans.fit_predict(X_transform)
    mutinfo0 = mutual_info_score(pred0,numerals)
    mutinfo1 = mutual_info_score(pred1,numerals)
    output = pred0, pred1, numerals, mutinfo0, mutinfo1
    with open('mlp_compare_hl5.pkl','w') as f:
        pickle.dump(output, f)
    with open('mlp_transform_mnist.pkl', 'w') as f:
        pickle.dump(X_transform, f)
