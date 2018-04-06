import numpy as np


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """

# Ensure that x1 and x2 are column vectors
#     x1 = x1.ravel()
#     x2 = x2.ravel()

# You need to return the following variables correctly.
    sim = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Fill in this function to return the similarity between x1
#               and x2 computed using a Gaussian kernel with bandwidth
#               sigma
#
#
    norm_diff = np.linalg.norm(x1 - x2)
    pow_pol = (-np.power(norm_diff,2))/(2.0*np.power(sigma,2))
    sim = np.power(np.e,pow_pol)

# =============================================================
    return sim