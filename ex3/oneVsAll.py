import numpy as np
from scipy.optimize import minimize,fmin_cg

from lrCostFunction import lrCostFunction
from ex2.gradientFunctionReg import gradientFunctionReg


def oneVsAll(X, y, num_labels, Lambda):
    """trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

# Some useful variables
    m, n = X.shape

# You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

# Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

# ====================== YOUR CODE HERE ======================
# Instructions: You should complete the following code to train num_labels
#               logistic regression classifiers with regularization
#               parameter lambda. 
#
# Hint: theta(:) will return a column vector.
#
# Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
#       whether the ground truth is true/false for this class.
#
# Note: For this assignment, we recommend using fmincg to optimize the cost
#       function. It is okay to use a for-loop (for c = 1:num_labels) to
#       loop over the different classes.

    # Set Initial theta
    initial_theta = np.zeros((n + 1))

    # This function will return theta and the cost
   # opts = {'maxiter' : None,    # default value.
   #      'disp' : False,    # non-default value.
   #      'gtol' : 1e-5,    # default value.
   #      'norm' : np.inf,  # default value.
   #      'eps' : 1.4901161193847656e-08}  # default value.
        
    for i in range(num_labels):
        #using L-BFGS-B algorithm
        res = minimize(lrCostFunction, initial_theta, method='L-BFGS-B',
               jac=gradientFunctionReg, args=(X, np.where(y==i+1,1,0), Lambda), options={'disp': False, 'maxiter': 26})
        all_theta[i] = res.x
        cost_t = res.fun
        print 'the %s th cost is : %0.4f' %(i+1,cost_t)
       #using fmin_cg algorithm

       # res = fmin_cg(lrCostFunction, initial_theta, fprime=gradientFunctionReg, \
       #               args=(X, np.where(y==i+1,1,0), Lambda),gtol=1e-15, maxiter=26,full_output=1)
       # all_theta[i] = res[0]
       # cost_t = res[1]
       # print 'the %s th cost is : %0.4f' %(i+1,cost_t)

# =========================================================================

    return all_theta

