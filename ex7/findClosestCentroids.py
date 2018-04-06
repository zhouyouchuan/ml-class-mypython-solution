import numpy as np


def findClosestCentroids(X, centroids):
    """returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """

# Set K
    K = len(centroids)

# You need to return the following variables correctly.
    idx = np.zeros(X.shape[0])

# ====================== YOUR CODE HERE ======================
# Instructions: Go over every example, find its closest centroid, and store
#               the index inside idx at the appropriate location.
#               Concretely, idx(i) should contain the index of the centroid
#               closest to example i. Hence, it should be a value in the 
#               range 1..K
#
# Note: You can use a for-loop over the examples to compute this.
    val = np.zeros(X.shape[0])

    length = len(X)
    for i in range(length):
        exchan_N = []
        for j in range(K):     
            norm_fan = np.linalg.norm(X[i,:] - centroids[j])
            norm_pow = np.power(norm_fan,2)
            exchan_N.append(norm_pow)
        val[i] = min(exchan_N)
        idx[i] = exchan_N.index(val[i])
# =============================================================

    return val, idx

