from numpy import asfortranarray, squeeze
from gradientFunction import gradientFunction
from sigmoid import sigmoid
import numpy as np

def gradientFunctionReg(theta, X, y, Lambda):
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
    ureg_grad = gradientFunction(theta, X, y) 
    grad = ureg_grad + (1.*Lambda/m) * squeeze(theta)
    tem_X = asfortranarray(X)
    tem_X = squeeze(tem_X)
    tem_y = asfortranarray(y) 
    tem_y = squeeze(y)
    #grad = np.zeros(theta.shape)    
    tem_h = sigmoid(np.dot(tem_X,theta))
    tem_h = squeeze(tem_h)
    
    sub_h = tem_h - tem_y
    #sub_h = sub_h.reshape(m,1)
    tem0 = (1./m) * np.sum(sub_h * tem_X[:,0],axis=0)
    grad[0] = tem0
    #grad = (1./m) * np.multiply(sub_h, tem_X)
   
# =============================================================

    return grad