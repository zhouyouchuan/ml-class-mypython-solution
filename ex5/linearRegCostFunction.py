import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#
    if theta.ndim <2:
        tem_theta = np.expand_dims(theta,axis=1)
    if y.ndim <2:
        tem_y = np.expand_dims(y,axis=1)
    hy = np.dot(X,tem_theta)
    err_dif = hy - tem_y
    reg_theta = np.power(theta[1:], 2)
    J = (1./(2.*m)) * np.sum(np.power(err_dif,2)) + (1.0*Lambda/(2*m)) * np.sum(reg_theta)
    grad_mul = err_dif * X
    grad0 = (1./m) * np.sum(err_dif )
    grad = (1./m) * np.sum(grad_mul,axis=0) + (1.0*Lambda/m) * theta
    grad[0] = grad0
    
# =========================================================================

    return J, grad