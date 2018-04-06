from sigmoid import sigmoid
from numpy import squeeze, asarray,reshape
import numpy as np

def gradientFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """

    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta        
    
    tem_X = asarray(X)
    tem_X = squeeze(tem_X)
    
    tem_y = asarray(y) 
    tem_y = squeeze(y)
    #grad = np.zeros(theta.shape)    
    tem_h = sigmoid(np.dot(tem_X,theta))
    tem_h = squeeze(tem_h)
    
    sub_h = tem_h - tem_y
    sub_h = sub_h.reshape(m,1)
    
    #grad = (1./m) * np.matmul(sub_h, tem_X)
    grad = (1./m) * np.sum(sub_h*X,axis=0)
   
# =============================================================

    return grad
