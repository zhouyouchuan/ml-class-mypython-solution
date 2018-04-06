from numpy import log
from sigmoid import sigmoid
import numpy as np
def costFunction(theta, X,y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

# Initialize some useful values
    #J = 0.    
    m = y.size # number of training examples
    
    #down dim
    tem_h = sigmoid(np.dot(X,theta))
    tem_h = np.squeeze(tem_h)
    
    h_one = log(tem_h)
    #h_zero = log(np.subtract(1, tem_h))
    h_zero = log(1 - tem_h)
    y = np.asarray(y)
    y = np.squeeze(y)
    # mul_one = np.multiply(-y, h_one)
    # mul_zero = np.multiply((1-y), h_zero)
    # scalar multiply = *
    mul_one = (-y) * h_one
    mul_zero = (1 - y) * h_zero
    
    J = (1./m) * np.sum(mul_one - mul_zero)

    
    
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta
#
# Note: grad should have the same dimensions as theta
#
    return J
